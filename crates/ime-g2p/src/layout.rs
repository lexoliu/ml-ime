//! Where each pipeline stage keeps its parquet shards under one data root.
//!
//! Nothing here knows an absolute path: the same commands run against a
//! checkout's `data/` and against a kernel's working directory, so every stage
//! is told a root and derives the rest.

use std::path::{Path, PathBuf};

/// One data root, and the directories the stages read and write under it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataLayout {
    root: PathBuf,
}

impl DataLayout {
    /// A layout rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root itself.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Normalised documents, straight off the network.
    #[must_use]
    pub fn documents(&self) -> PathBuf {
        self.root.join("documents")
    }

    /// Filtered target sentences with their preceding context.
    #[must_use]
    pub fn samples(&self) -> PathBuf {
        self.root.join("samples")
    }

    /// Dual g2p annotations, plus the hard and refused subsets.
    #[must_use]
    pub fn annotations(&self) -> PathBuf {
        self.root.join("annotations")
    }

    /// The pool file a run's samples were drawn into.
    #[must_use]
    pub fn pool(&self) -> PathBuf {
        self.root.join("pool.jsonl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage_hangs_off_the_root_it_was_given() {
        let layout = DataLayout::new("data/run1_pool");
        assert_eq!(layout.samples(), Path::new("data/run1_pool/samples"));
        assert_eq!(
            layout.annotations(),
            Path::new("data/run1_pool/annotations")
        );
        assert_eq!(layout.pool(), Path::new("data/run1_pool/pool.jsonl"));
    }
}
