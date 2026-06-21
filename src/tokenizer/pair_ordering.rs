use crate::Token;

#[derive(Debug, Eq)]
pub(crate) struct PairOrd(pub u32, pub (Token, Token));
impl PairOrd {
    fn len(&self) -> usize {
        self.0 as usize
    }
}

impl PartialEq for PairOrd {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl PartialOrd for PairOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len()
            .cmp(&other.len())
            .then_with(|| self.1.cmp(&other.1))
    }
}
