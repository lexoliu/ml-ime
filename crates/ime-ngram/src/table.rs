//! A read-only map from a packed n-gram key to a probability.

use std::ops::Range;
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeStruct};

use crate::NgramError;

/// Where a table's two arrays live: owned on the heap, or ranges of a
/// memory-mapped file the model was loaded from.
#[derive(Clone, Debug)]
enum Storage {
    /// Two boxes, the shape a trained table takes.
    Owned {
        /// Sorted keys.
        keys: Box<[u64]>,
        /// The probability under each key.
        values: Box<[f32]>,
    },
    /// Byte ranges of one shared mapping: keys as little-endian `u64`s, values
    /// as little-endian `f32`s. Nothing is copied out until a lookup touches
    /// it, and the pages stay shared with the file cache.
    Mapped {
        /// The file the ranges point into.
        map: Arc<Mmap>,
        /// Byte range of the key array.
        keys: Range<usize>,
        /// Byte range of the value array.
        values: Range<usize>,
    },
}

/// Sorted keys beside their values, looked up by binary search.
///
/// A hash map would be a wash on lookup and much worse on both file size and
/// load: this is two flat arrays, so deserialising it is two length-prefixed
/// reads and no hashing at all. The arrays may also sit in a memory-mapped
/// file rather than the heap, which is how a large model shares its pages
/// with the file cache instead of owning them.
#[derive(Clone, Debug)]
pub struct ProbTable {
    storage: Storage,
}

/// The wire shape the derive would have produced; serialisation keeps it so
/// files written before the mapped storage existed still load, and a mapped
/// table serialises back to owned bytes.
#[derive(Serialize, Deserialize)]
struct Wire {
    keys: Box<[u64]>,
    values: Box<[f32]>,
}

impl ProbTable {
    /// Build a table from unordered `(key, value)` pairs.
    ///
    /// # Panics
    ///
    /// If *entries* holds the same key twice.
    #[must_use]
    pub fn build(mut entries: Vec<(u64, f32)>) -> Self {
        entries.sort_unstable_by_key(|(key, _)| *key);
        assert!(
            entries.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "n-gram table was built with a duplicate key"
        );
        let (keys, values) = entries.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        Self::owned(keys.into_boxed_slice(), values.into_boxed_slice())
    }

    /// Wrap two owned arrays.
    pub(crate) fn owned(keys: Box<[u64]>, values: Box<[f32]>) -> Self {
        Self {
            storage: Storage::Owned { keys, values },
        }
    }

    /// Point a table at byte ranges of a shared mapping.
    ///
    /// # Errors
    ///
    /// If the ranges run off the map, are misaligned for their element types,
    /// or hold different numbers of keys and values.
    pub(crate) fn mapped(
        map: Arc<Mmap>,
        keys: Range<usize>,
        values: Range<usize>,
    ) -> Result<Self, NgramError> {
        let key_len = keys.end - keys.start;
        let value_len = values.end - values.start;
        if !key_len.is_multiple_of(size_of::<u64>())
            || !value_len.is_multiple_of(size_of::<f32>())
            || key_len / size_of::<u64>() != value_len / size_of::<f32>()
        {
            return Err(NgramError::Corrupt);
        }
        if !keys.start.is_multiple_of(align_of::<u64>())
            || !values.start.is_multiple_of(align_of::<f32>())
            || keys.end > map.len()
            || values.end > map.len()
        {
            return Err(NgramError::Corrupt);
        }
        Ok(Self {
            storage: Storage::Mapped { map, keys, values },
        })
    }

    /// The sorted key array.
    pub(crate) fn keys(&self) -> &[u64] {
        match &self.storage {
            Storage::Owned { keys, .. } => keys,
            Storage::Mapped { map, keys, .. } => bytemuck::cast_slice(&map[keys.clone()]),
        }
    }

    /// The value array parallel to [`ProbTable::keys`].
    pub(crate) fn values(&self) -> &[f32] {
        match &self.storage {
            Storage::Owned { values, .. } => values,
            Storage::Mapped { map, values, .. } => bytemuck::cast_slice(&map[values.clone()]),
        }
    }

    /// The value stored under *key*, if any.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<f32> {
        self.keys()
            .binary_search(&key)
            .ok()
            .map(|index| self.values()[index])
    }

    /// A forward cursor over the keys, started at the first entry ≥ *key*.
    ///
    /// A scan of sorted keys that share a key-space row walks forward from
    /// here: each [`Row::at`] step costs adjacent entries where a fresh
    /// `get` pays a whole binary search — over a memory-mapped table, whole
    /// cold pages.
    pub(crate) fn row(&self, key: u64) -> Row<'_> {
        Row {
            keys: self.keys(),
            values: self.values(),
            at: self.keys().partition_point(|&stored| stored < key),
        }
    }

    /// How many entries the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys().len()
    }
}

/// A forward-only merge cursor over a [`ProbTable`]'s sorted keys.
///
/// `at` asks for keys in non-decreasing order; each probe advances the
/// cursor instead of restarting a binary search, so a beam's sorted
/// candidate list resolves in one walk of the key-space row.
pub(crate) struct Row<'a> {
    /// The table's sorted keys.
    keys: &'a [u64],
    /// Its parallel values.
    values: &'a [f32],
    /// The first index the next key could occupy.
    at: usize,
}

impl Row<'_> {
    /// Advance to *key* — which must be ≥ the last asked — and return its
    /// value if the table holds it.
    pub(crate) fn at(&mut self, key: u64) -> Option<f32> {
        while self.at < self.keys.len() && self.keys[self.at] < key {
            self.at += 1;
        }
        (self.at < self.keys.len() && self.keys[self.at] == key).then(|| self.values[self.at])
    }
}

impl Default for ProbTable {
    fn default() -> Self {
        Self::owned(Box::default(), Box::default())
    }
}

impl Serialize for ProbTable {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("ProbTable", 2)?;
        state.serialize_field("keys", self.keys())?;
        state.serialize_field("values", self.values())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ProbTable {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(Self::owned(wire.keys, wire.values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_finds_what_was_stored_and_nothing_else() {
        let table = ProbTable::build(vec![(7, 0.5), (2, 0.25), (9, 0.125)]);
        assert_eq!(table.get(2), Some(0.25));
        assert_eq!(table.get(7), Some(0.5));
        assert_eq!(table.get(9), Some(0.125));
        assert_eq!(table.get(8), None);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn an_empty_table_finds_nothing() {
        let table = ProbTable::build(Vec::new());
        assert_eq!(table.len(), 0);
        assert_eq!(table.get(0), None);
    }

    #[test]
    fn a_serialised_table_reads_back_identically() {
        let table = ProbTable::build(vec![(7, 0.5), (2, 0.25), (9, 0.125)]);
        let bytes = postcard::to_stdvec(&table).expect("the table serialises");
        let reloaded: ProbTable = postcard::from_bytes(&bytes).expect("the table reloads");
        assert_eq!(reloaded.len(), 3);
        assert_eq!(reloaded.get(2), Some(0.25));
        assert_eq!(reloaded.get(7), Some(0.5));
        assert_eq!(reloaded.get(9), Some(0.125));
    }
}
