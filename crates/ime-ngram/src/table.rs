//! A read-only map from a packed n-gram key to a probability.

use serde::{Deserialize, Serialize};

/// Sorted keys beside their values, looked up by binary search.
///
/// A hash map would be a wash on lookup and much worse on both file size and
/// load: this is two flat arrays, so deserialising it is two length-prefixed
/// reads and no hashing at all.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProbTable {
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
        Self {
            keys: keys.into_boxed_slice(),
            values: values.into_boxed_slice(),
        }
    }

    /// The value stored under *key*, if any.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<f32> {
        self.keys
            .binary_search(&key)
            .ok()
            .map(|index| self.values[index])
    }

    /// How many entries the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
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
}
