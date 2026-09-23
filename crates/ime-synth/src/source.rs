//! Where a seed term came from, and what the synthetic samples are labelled with.
//!
//! The seed sources are listed in the priority order the decision record sets:
//! the manually reviewed Sogou dictionary first, the unreviewed bilibili one
//! second, then the two sources that carry explanations -- 梗百科 and the
//! wikipedia slang list -- which seed the terms the dictionaries do not have. A
//! term that appears in more than one is attributed to the first that carried it,
//! so the provenance sidecar has exactly one row per generated sample and a bad
//! dictionary can be dropped by filtering on `seed_source` alone.
//!
//! Attribution and grounding are two different questions and the sidecar answers
//! both. A Sogou word is *attributed* to Sogou however it was explained, and the
//! `grounding` column says which source the explanation in its prompt came from,
//! so a batch can be dropped either by the dictionary that proposed the term or
//! by the encyclopaedia that described it.

/// The value of the `source` column on every sample this crate writes, and the
/// shard prefix they are written under.
///
/// It is deliberately not one of the corpus source names: these sentences are
/// training-only, and the evaluation draw must be able to exclude them by name.
pub const SYNTHETIC: &str = "synthetic-luna";

/// The shard prefix the provenance sidecar is written under.
pub const PROVENANCE: &str = "provenance";

/// One seed lexicon: what it is called in the shards, and where it is read from.
///
/// `slug` is the parquet prefix under the lexicon directory, and is `None` for
/// the wikipedia list because that one arrives as a `MediaWiki` parse response
/// rather than as lexicon shards.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SeedSource {
    /// The value written into the provenance sidecar's `seed_source` column.
    pub name: &'static str,
    /// The Sogou dictionary id, for the two that have one.
    pub dict_id: Option<i64>,
    /// The lexicon shard prefix, for the two that are lexicon shards.
    pub slug: Option<&'static str>,
}

/// 网络流行新词 (id 4): manually reviewed, and the highest-priority seed.
pub const SOGOU_PREMIUM: SeedSource = SeedSource {
    name: "sogou-premium",
    dict_id: Some(4),
    slug: Some("wangluo-liuxing-xinci"),
};

/// 哔哩网梗 (id 177287): unreviewed, tagged so it can be filtered back out.
pub const SOGOU_BILIBILI: SeedSource = SeedSource {
    name: "sogou-bilibili",
    dict_id: Some(177_287),
    slug: Some("bilibili-wanggeng"),
};

/// 梗百科 (gengbaike.cn): 2.3k crawled 梗 articles, each one an explanation.
pub const GENGBAIKE: SeedSource = SeedSource {
    name: "gengbaike",
    dict_id: None,
    slug: None,
};

/// The zh-wikipedia 中国大陆网络用语列表, the other seed that carries explanations.
pub const WIKI_SLANG: SeedSource = SeedSource {
    name: "wiki-slang",
    dict_id: None,
    slug: None,
};

/// Every seed source, in the priority order a term is attributed by.
pub const SEED_SOURCES: [SeedSource; 4] = [SOGOU_PREMIUM, SOGOU_BILIBILI, GENGBAIKE, WIKI_SLANG];

/// Every source that carries explanations, in the order a term is grounded by.
///
/// 梗百科 comes first because it explains 梗 at article length where the
/// wikipedia list gives a sentence, and because it covers far more of the Sogou
/// dictionaries than the list does.
pub const GROUNDING_SOURCES: [SeedSource; 2] = [GENGBAIKE, WIKI_SLANG];
