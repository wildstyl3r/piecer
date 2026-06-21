use std::borrow::Borrow;

use crate::TokenType;

pub(crate) struct ArenaNode<T> {
    pub value: T,
    prev: Option<usize>,
    next: Option<usize>,
}

pub(crate) struct ArenaList<T>(Vec<ArenaNode<T>>);

impl<T: TokenType> ArenaList<T> {
    fn last_mut(&mut self) -> Option<&mut ArenaNode<T>> {
        self.0.last_mut()
    }

    pub fn raw_pairs(&self) -> std::slice::Windows<'_, ArenaNode<T>> {
        self.0.windows(2)
    }

    pub fn drop(&mut self, index: usize) -> bool {
        if index > self.0.len() - 1 {
            false
        } else {
            let (p, n) = (self.0[index].prev, self.0[index].next);
            self.0[index].prev = None;
            self.0[index].next = None;
            if let Some(prev_index) = p {
                self.0[prev_index].next = n
            }
            if let Some(next_index) = n {
                self.0[next_index].prev = p
            }
            true
        }
    }

    pub fn pair_at(&self, first: usize) -> Option<(T, T)> {
        if first >= self.0.len() - 1 {
            None
        } else {
            self.0[first]
                .next
                .map(|second| (self.0[first].value, self.0[second].value))
        }
    }

    pub fn prev_pair_pos(&self, second: usize) -> Option<((T, T), usize)> {
        if second > self.0.len() - 1 {
            None
        } else {
            self.0[second]
                .prev
                .map(|first| ((self.0[first].value, self.0[second].value), first))
        }
    }

    pub fn next_pair_pos(&self, proto: usize) -> Option<((T, T), usize)> {
        if proto >= self.0.len() - 1 {
            None
        } else {
            match self.0[proto].next {
                Some(first) => self.0[first]
                    .next
                    .map(|second| ((self.0[first].value, self.0[second].value), first)),
                None => None,
            }
        }
    }

    pub fn fuse_into(
        &mut self,
        index: usize,
        tok: T,
    ) -> (
        Option<((T, T), usize)>,
        Option<((T, T), usize)>,
    ) {
        match self.0[index].next {
            Some(second) => {
                self.0[index].value = tok;
                self.drop(second);
                (
                    self.prev_pair_pos(index),
                    self.pair_at(index).map(|p| (p, index)),
                )
            }
            None => panic!("pair fusion failed"),
        }
    }
}

impl<I, T> FromIterator<I> for ArenaList<T>
where
    I: Borrow<T>,
    T: TokenType,
{
    fn from_iter<It: IntoIterator<Item = I>>(iter: It) -> Self {
        let mut al = ArenaList(Vec::new());
        for (i, value) in iter.into_iter().enumerate() {
            al.0.push(ArenaNode {
                value: *value.borrow(),
                prev: if i > 0 { Some(i - 1) } else { None },
                next: Some(i + 1),
            });
        }
        al.last_mut().unwrap().next = None;
        al
    }
}
