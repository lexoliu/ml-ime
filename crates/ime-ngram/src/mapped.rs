//! The memory-mappable layout [`NgramModel::open`] reads.
//!
//! A postcard file is a sequence of varints, so nothing in it can be sliced
//! into arrays without decoding the whole model into the heap first -- the
//! trigram alone costs most of a gigabyte of anonymous memory that way. This
//! layout is the opposite trade: a fixed header of `(length, offset)` pairs
//! naming little-endian arrays at 8-byte alignment, so the model's tables can
//! point straight into the mapping. The file is a little larger than the
//! postcard encoding and none of it is resident until a lookup touches a
//! page.
//!
//! ```text
//! offset 0    'M' 'G' 'N' 'M'          magic
//! offset 4    u32 = 1                  format version
//! offset 8    9 x { u64 len, u64 off } section descriptors
//! offset 152  sections, each 8-byte aligned:
//!             vocabulary       u32[]   lexicon-order code points
//!             unigram          f32[]   interpolated unigram level
//!             bigram_backoff   f32[]   bigram level's backoff weight
//!             bigram           u64[]   keys, then
//!             bigram           f32[]   values; the two trigram tables
//!             trigram_backoff  u64[]+f32[]  follow in the same order
//!             trigram          u64[]+f32[]
//! ```

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::NgramError;
use crate::model::NgramModel;
use crate::table::ProbTable;

/// Bytes a mapped model starts with.
const MAGIC: &[u8; 4] = b"MGNM";
/// The only layout version this reader knows.
const VERSION: u32 = 1;
/// How many section descriptors the header holds.
const SECTIONS: usize = 9;
/// Header size: magic + version + the descriptors.
const HEADER: usize = 8 + SECTIONS * 16;
/// Element size of each section, in descriptor order.
const ELEMENTS: [usize; SECTIONS] = [4, 4, 4, 8, 4, 8, 4, 8, 4];

/// A `(length, offset)` pair, little-endian, as the header stores it.
#[derive(Debug, Clone, Copy)]
struct Section {
    /// Element count.
    len: usize,
    /// Byte offset the array starts at.
    offset: usize,
}

impl Section {
    /// The byte range the array occupies.
    fn range(&self, element: usize) -> Range<usize> {
        self.offset..self.offset + self.len * element
    }
}

/// Whether the file starts with the mapped format's magic.
pub(crate) fn is_mapped(path: &Path) -> bool {
    let mut probe = [0_u8; 4];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut probe))
        .is_ok_and(|()| &probe == MAGIC)
}

/// Append *values* and return the section that describes where they landed;
/// every section starts 8-aligned.
fn write_section<T: bytemuck::Pod>(out: &mut Vec<u8>, values: &[T]) -> Section {
    let pad = (8 - out.len() % 8) % 8;
    out.resize(out.len() + pad, 0);
    let section = Section {
        len: values.len(),
        offset: out.len(),
    };
    out.extend_from_slice(bytemuck::cast_slice(values));
    section
}

/// Serialise a model into the mapped layout.
///
/// # Errors
///
/// If the file cannot be written.
pub(crate) fn write(model: &NgramModel, path: &Path) -> Result<(), NgramError> {
    let vocabulary: Vec<u32> = model.vocabulary().iter().map(|ch| *ch as u32).collect();
    let mut body = vec![0_u8; HEADER];
    let mut sections = Vec::with_capacity(SECTIONS);
    sections.push(write_section(&mut body, &vocabulary));
    sections.push(write_section(&mut body, model.unigram()));
    sections.push(write_section(&mut body, model.bigram_backoff()));
    for table in model.tables() {
        sections.push(write_section(&mut body, table.keys()));
        sections.push(write_section(&mut body, table.values()));
    }
    debug_assert_eq!(sections.len(), SECTIONS);
    let mut header = Vec::with_capacity(HEADER);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    for section in &sections {
        header.extend_from_slice(&(section.len as u64).to_le_bytes());
        header.extend_from_slice(&(section.offset as u64).to_le_bytes());
    }
    body[..HEADER].copy_from_slice(&header);
    let file = File::create(path).map_err(NgramError::Io)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&body).map_err(NgramError::Io)?;
    writer.flush().map_err(NgramError::Io)
}

/// Read one descriptor pair out of the header; the caller checked the map is
/// at least [`HEADER`] bytes.
fn section(map: &Mmap, index: usize) -> Section {
    let at = 8 + index * 16;
    let len = u64::from_le_bytes(
        map[at..at + 8]
            .try_into()
            .unwrap_or_else(|_| unreachable!("the slice is eight bytes")),
    );
    let offset = u64::from_le_bytes(
        map[at + 8..at + 16]
            .try_into()
            .unwrap_or_else(|_| unreachable!("the slice is eight bytes")),
    );
    Section {
        len: usize::try_from(len).unwrap_or(usize::MAX),
        offset: usize::try_from(offset).unwrap_or(usize::MAX),
    }
}

/// A slice of the map reinterpreted as `T`s; the caller's section is
/// 8-aligned and sized in whole elements inside the map, so the cast and the
/// slice cannot fail.
fn cast<T: bytemuck::Pod>(map: &Mmap, range: Range<usize>) -> &[T] {
    bytemuck::cast_slice(&map[range])
}

/// Map *file* read-only and load the model over it, checking the header and
/// the lexicon.
///
/// # Errors
///
/// The same set [`load`] reports, plus the file open and map failures.
pub(crate) fn open_file(
    path: &Path,
    lexicon: &ime_pinyin::Lexicon,
) -> Result<NgramModel, NgramError> {
    let file = File::open(path).map_err(NgramError::Io)?;
    // The file is trusted: mutating a read-only private map's pages from
    // another process would be a data race outside this program's control,
    // which the loader accepts the same way every weights mmap in `ime-lm`
    // does.
    let map = unsafe { Mmap::map(&file) }.map_err(NgramError::Io)?;
    load(map, lexicon)
}

/// Load a model from its mapping, checking the header and the lexicon.
///
/// # Errors
///
/// If the file is truncated, names a version this reader does not know, holds
/// sections outside its own extent or misaligned for their element type, or
/// fails the same lexicon check [`NgramModel::from_bytes`] applies.
pub(crate) fn load(map: Mmap, lexicon: &ime_pinyin::Lexicon) -> Result<NgramModel, NgramError> {
    if map.len() < HEADER
        || map[0..4] != MAGIC[..]
        || u32::from_le_bytes(map[4..8].try_into().unwrap_or_default()) != VERSION
    {
        return Err(NgramError::Corrupt);
    }
    for (index, &element) in ELEMENTS.iter().enumerate() {
        let section = section(&map, index);
        let bytes = section.len.saturating_mul(element);
        if !section.offset.is_multiple_of(8)
            || section.offset > map.len()
            || bytes > map.len() - section.offset
        {
            return Err(NgramError::Corrupt);
        }
    }
    let map = Arc::new(map);
    let at = |index: usize| section(&map, index);
    let vocabulary = cast::<u32>(&map, at(0).range(ELEMENTS[0]))
        .iter()
        .map(|&code| char::from_u32(code).ok_or(NgramError::Corrupt))
        .collect::<Result<Box<[char]>, _>>()?;
    let unigram = cast::<f32>(&map, at(1).range(ELEMENTS[1])).to_vec();
    let bigram_backoff = cast::<f32>(&map, at(2).range(ELEMENTS[2])).to_vec();
    let table = |key_index: usize, value_index: usize| -> Result<ProbTable, NgramError> {
        ProbTable::mapped(
            map.clone(),
            at(key_index).range(ELEMENTS[key_index]),
            at(value_index).range(ELEMENTS[value_index]),
        )
    };
    let model = NgramModel::new(
        vocabulary,
        unigram.into_boxed_slice(),
        bigram_backoff.into_boxed_slice(),
        table(3, 4)?,
        table(5, 6)?,
        table(7, 8)?,
    );
    model.check_lexicon(lexicon)?;
    Ok(model)
}
