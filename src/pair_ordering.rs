use std::collections::HashSet;

use crate::Token;

#[derive(Debug, Eq)]
pub struct StableOrdHashSet(pub HashSet<(usize, usize)>, pub (Token, Token), pub u32);
impl StableOrdHashSet {
    fn len(&self) -> usize {
        self.0.len() * self.2 as usize
    }
}

impl PartialEq for StableOrdHashSet {
    fn eq(&self, other: &Self) -> bool {
        (self.len() == other.len()) && (self.1 == other.1)
    }
}

impl PartialOrd for StableOrdHashSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StableOrdHashSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len()
            .cmp(&other.len())
            .then_with(|| self.1.cmp(&other.1))
    }
}
