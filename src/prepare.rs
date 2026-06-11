use std::sync::OnceLock;

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

    let mut insertion_counter = 0;
    let mut prev_type = GroupType::Other;
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
        if (letter && prev_type != GroupType::Letters) || (digit && prev_type != GroupType::Digits)
        {
            insertion_counter += 1;
        }
        if letter {
            prev_type = GroupType::Letters;
        } else if digit {
            prev_type = GroupType::Digits;
        } else if c == ' ' || c == '▁' {
        } else {
            prev_type = GroupType::Other;
        }
    }

    let mut result = String::with_capacity(s.len() + insertion_counter);

    prev_type = GroupType::Other;
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

pub(crate) fn chunks(s: &str) -> impl Iterator<Item = &str> {
    let pattern = r"'(?:[stmd]|re|ve|ll)|▁(\p{L}+(?:[\p{L}-]*\p{L})?|\p{N}{1,3})|\p{N}{1,3}|[^\s\p{L}\p{N}▁]+|\s+";
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(pattern).unwrap());

    re.find_iter(s).map(|mat| mat.as_str())
}
