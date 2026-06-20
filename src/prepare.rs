use std::sync::OnceLock;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use regex::Regex;

#[derive(Clone, Copy, PartialEq)]
enum GroupType {
    Letters,
    Digits,
    Other,
}

pub(crate) fn normalize(s: &str) -> String {
    // let re = Regex::new(
    //     r"(?P<L>\p{L}+(?:-\p{L}+)*)|(?P<D>\p{N}+)|(?P<S>[▁ ]+)|(?P<O>[^\p{L}\p{N} ▁]+)",
    // )
    // .unwrap();
    let mut result = String::with_capacity(s.len());

    let mut prev_type = GroupType::Other;
    let mut prev_char = None;
    let mut iter = s.chars().peekable();
    while let Some(c) = iter.next() {
        let letter = c.is_alphabetic()
            || (c == '-' && {
                if let Some(next_c) = iter.peek() {
                    next_c.is_alphabetic() && (prev_type == GroupType::Letters)
                } else {
                    false
                }
            });
        let digit = c.is_numeric();
        if ((letter && prev_type != GroupType::Letters)
            || (digit && prev_type != GroupType::Digits))
            && (prev_char != Some('▁'))
        {
            result.push('▁');
        }
        if c == ' ' || c == '▁' {
            if let Some(next_c) = iter.peek() {
                let letter_next = next_c.is_alphabetic();
                let digit_next = next_c.is_numeric();
                if (letter_next && prev_type == GroupType::Letters)
                    || (digit_next && prev_type == GroupType::Digits)
                {
                    result.push('▁');
                } else {
                    result.push(c);
                }
            }
        } else {
            if letter {
                prev_type = GroupType::Letters;
            } else if digit {
                prev_type = GroupType::Digits;
            } else {
                prev_type = GroupType::Other;
            }
            result.push(c);
        }
        prev_char = Some(c);
    }
    result
}

pub(crate) fn denormalize(s: &str) -> String {
    // regex equivalent:
    // Regex::new(r"(^|[^\p{L}] *)▁(?=\p{L})|(^|[^\p{N}] *)▁(?=\p{N})")
    //     .unwrap()
    //     .replace_all(&Regex::new(r"(\p{L} *)▁(?=\p{L})|(\p{N} *)▁(?=\p{N})")
    //     .unwrap()
    //     .replace_all(s, "$1$2 "), "$1$2")
    let mut result = String::with_capacity(s.len());
    let mut prev_type = GroupType::Other;
    let mut pref_flag = false;
    for c in s.chars() {
        if c == '▁' {
            pref_flag = true;
        } else {
            let mut replace_mode: bool = false;
            if c.is_alphabetic() {
                if prev_type != GroupType::Letters {
                    prev_type = GroupType::Letters;
                } else {
                    replace_mode = true;
                }
            } else if c.is_numeric() {
                if prev_type != GroupType::Digits {
                    prev_type = GroupType::Digits;
                } else {
                    replace_mode = true;
                }
            } else if c != ' ' && prev_type != GroupType::Other {
                prev_type = GroupType::Other;
            }

            if pref_flag && replace_mode {
                result.push(' ');
            }
            pref_flag = false;
            result.push(c);
        }
    }
    result
}

enum Chunk<'a> {
    Whitespace(&'a str),
    Text(&'a str),
}

enum LocalIter<'a, I> {
    Once(std::iter::Once<&'a str>),
    Regex(I),
}

impl<'a, I> Iterator for LocalIter<'a, I>
where
    I: Iterator<Item = &'a str>,
{
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LocalIter::Once(iter) => iter.next(),
            LocalIter::Regex(iter) => iter.next(),
        }
    }
}

pub(crate) fn chunks(s: &str) -> Vec<&str> {
    //impl Iterator<Item = &str> {
    // let pattern = r"'(?:[stmd]|re|ve|ll)|▁(\p{L}+(?:[\p{L}-]*\p{L})?|\p{N}{1,3})|\p{N}{1,3}|[^\s\p{L}\p{N}▁]+|\s+";
    // static RE: OnceLock<Regex> = OnceLock::new();
    // let re = RE.get_or_init(|| Regex::new(pattern).unwrap());

    // re.find_iter(s).map(|mat| mat.as_str())
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        let pattern = r"(?:[stmd]|re|ve|ll)|▁(\p{L}+(?:[\p{L}-]*\p{L})?|\p{N}{1,3})|\p{N}{1,3}|[^\s\p{L}\p{N}▁]+";
        Regex::new(pattern).unwrap()
    });

    let mut chunks = Vec::new();
    let mut last_idx = 0;
    let mut in_whitespace = None;

    for (idx, ch) in s.char_indices() {
        let is_ws = ch.is_whitespace();
        if let Some(ws) = in_whitespace {
            if ws != is_ws {
                let slice = &s[last_idx..idx];
                if ws {
                    chunks.push(Chunk::Whitespace(slice));
                } else {
                    chunks.push(Chunk::Text(slice));
                }
                last_idx = idx;
                in_whitespace = Some(is_ws);
            }
        } else {
            in_whitespace = Some(is_ws);
        }
    }

    if last_idx < s.len() {
        let slice = &s[last_idx..];
        if in_whitespace.unwrap_or(false) {
            chunks.push(Chunk::Whitespace(slice));
        } else {
            chunks.push(Chunk::Text(slice));
        }
    }

    chunks
        .into_par_iter()
        .flat_map_iter(|chunk| match chunk {
            Chunk::Whitespace(ws) => LocalIter::Once(std::iter::once(ws)),
            Chunk::Text(txt) => {
                let iter = re.find_iter(txt).map(|mat| mat.as_str());
                LocalIter::Regex(iter)
            }
        })
        .collect()
}
