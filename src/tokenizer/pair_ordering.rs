use crate::TokenType;

#[derive(Debug, Eq)]
pub(crate) struct PairOrd<T: TokenType>(pub u32, pub (T, T));

impl<T: TokenType> PairOrd<T> {
    fn len(&self) -> usize {
        self.0 as usize
    }
}

impl<T: TokenType> PartialEq for PairOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<T: TokenType> PartialOrd for PairOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: TokenType> Ord for PairOrd<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len()
            .cmp(&other.len())
            .then_with(|| self.1.cmp(&other.1))
    }
}
