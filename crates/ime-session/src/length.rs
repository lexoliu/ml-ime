//! The session's choice between two engines by the length of the pending
//! input.

use crate::{Engine, Pending, Sentence};
use thiserror::Error;

/// An engine that reads input only up to a number of keys.
pub trait Bounded {
    /// The most letters the engine reads; input of more is outside it.
    fn max_keys(&self) -> usize;
}

/// Why a [`ByLength`] call failed: the engine that answered failed.
#[derive(Debug, Error)]
pub enum ByLengthError<S: std::error::Error + 'static, L: std::error::Error + 'static> {
    /// The short-input engine failed.
    #[error(transparent)]
    Short(S),
    /// The long-input engine failed.
    #[error(transparent)]
    Long(L),
}

/// An [`Engine`] that sends the pending input to one of two engines by its
/// length.
///
/// Input of at most [`Bounded::max_keys`] letters of the first engine, the
/// short-input engine, goes to it; longer input, which the first engine
/// cannot read, goes to the second. The limit is the first engine's own, and
/// the count is the pending letters, `'` boundaries not counted. Each call
/// reaches exactly one engine, and [`Engine::clear`] reaches both.
pub struct ByLength<S, L> {
    short: S,
    long: L,
}

impl<S: Engine + Bounded, L: Engine> ByLength<S, L> {
    /// An engine answering with *short* up to its limit and with *long*
    /// beyond it.
    #[must_use]
    pub const fn new(short: S, long: L) -> Self {
        Self { short, long }
    }

    /// The short-input engine.
    #[must_use]
    pub const fn short(&self) -> &S {
        &self.short
    }

    /// The long-input engine.
    #[must_use]
    pub const fn long(&self) -> &L {
        &self.long
    }
}

impl<S: Engine + Bounded, L: Engine> Engine for ByLength<S, L> {
    type Error = ByLengthError<S::Error, L::Error>;

    fn sentences(
        &mut self,
        pending: &Pending,
        context: Option<&str>,
    ) -> Result<Vec<Sentence>, Self::Error> {
        if pending.letters().len() <= self.short.max_keys() {
            self.short
                .sentences(pending, context)
                .map_err(ByLengthError::Short)
        } else {
            self.long
                .sentences(pending, context)
                .map_err(ByLengthError::Long)
        }
    }

    fn clear(&mut self) {
        self.short.clear();
        self.long.clear();
    }
}
