use std::collections::HashSet;
use std::hash::Hash;

use crate::Token;

#[derive(Debug, Eq)]
pub struct StableOrdHashSet<T: Hash>(pub HashSet<T>, pub (Token, Token));

impl<T: Hash> PartialEq for StableOrdHashSet<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.0.len() == other.0.len()) && (self.1 == other.1)
    }
}

impl<T: Hash + Eq> PartialOrd for StableOrdHashSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Hash + Eq> Ord for StableOrdHashSet<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.1.cmp(&other.1))
    }
}
