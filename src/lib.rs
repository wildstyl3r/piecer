use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use fancy_regex::{Captures, Regex};
use priority_queue::PriorityQueue;

pub type Token = u16;

pub struct Tokenizer {
    str2token: HashMap<String, Token>,
    token2str: Vec<String>,
    longest_str: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum GroupType {
    Letters,
    Digits,
    Spaces,
    Other,
}

struct ArenaNode {
    value: Token,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Eq)]
pub struct OrdHashSet<T: Hash>(pub HashSet<T>);

impl<T: Hash> PartialEq for OrdHashSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
    }
}

impl<T: Hash + Eq> PartialOrd for OrdHashSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Hash + Eq> Ord for OrdHashSet<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.len().cmp(&other.0.len())
    }
}

impl Tokenizer {
    pub fn train(s: &str, vocab_size: Option<usize>) -> Self {
        let mut alphabet = s
            .chars()
            .fold(HashSet::<String>::new(), |mut acc, c| {
                acc.insert(c.to_string());
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();
        alphabet.extend(["▁", "[UNK]", "[PAD]", "[BOS]", "[EOS]"].map(|s| s.to_string()));
        alphabet.sort();

        let mut tokenizer = Self {
            str2token: alphabet
                .iter()
                .enumerate()
                .map(|(i, t)| (t.to_owned(), i as Token))
                .collect(),
            longest_str: alphabet.iter().fold(0, |l, s| std::cmp::max(l, s.len())),
            token2str: alphabet,
        };
        println!("tok core done");

        if let Some(vocab_size) = vocab_size {
            let start = std::time::Instant::now();
            let norm = Tokenizer::normalize(s);
            println!("normalized in {:?}", std::time::Instant::now() - start);
            let mut chunks: Vec<Vec<ArenaNode>> = Tokenizer::chunks(&norm)
                .iter()
                .map(|str_chunk| {
                    let mut chunk: Vec<ArenaNode> = tokenizer
                        .encode_normalized(str_chunk)
                        .iter()
                        .enumerate()
                        .map(|(i, &tok)| ArenaNode {
                            value: tok,
                            prev: if i > 0 { Some(i - 1) } else { None },
                            next: Some(i + 1),
                        })
                        .collect();
                    chunk.last_mut().unwrap().next = None;
                    chunk
                })
                .collect();
            println!("chunking done");

            let mut bootstrap_counts = HashMap::new();
            for (chunk, chunk_v) in chunks.iter().enumerate() {
                for (i, pair_first) in chunk_v[..chunk_v.len() - 1]
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.next.is_some())
                {
                    let current_pair = (
                        pair_first.value,
                        chunk_v[*pair_first.next.as_ref().unwrap()].value,
                    );
                    bootstrap_counts
                        .entry(current_pair)
                        .or_insert(HashSet::new())
                        .insert((chunk, i));
                }
            }
            println!("count construction done");
            let mut pq_counts = PriorityQueue::new();
            for (k, v) in bootstrap_counts {
                pq_counts.push(k, OrdHashSet(v));
            }
            while tokenizer.token2str.len() < vocab_size {
                match pq_counts.peek() {
                    Some(((pair1, pair2), positions_set)) => {
                        let merge_str = tokenizer.token2str[*pair1 as usize].to_string()
                            + &tokenizer.token2str[*pair2 as usize];
                        let token = tokenizer.token2str.len() as Token;
                        tokenizer.longest_str =
                            std::cmp::max(tokenizer.longest_str, merge_str.len());
                        tokenizer.token2str.push(merge_str.clone());
                        tokenizer.str2token.insert(merge_str, token);

                        let mut positions: Vec<_> = positions_set.0.iter().collect();
                        positions.sort_by_key(|(_chunk, i)| i);
                        let mut decrements: HashMap<(u16, u16), HashSet<(usize, usize)>> =
                            HashMap::new();
                        let mut addons: HashMap<(u16, u16), HashSet<(usize, usize)>> =
                            HashMap::new();
                        for (c, first_index) in positions.into_iter().rev() {
                            if let Some(second_index) = chunks[*c][*first_index].next {
                                if let Some((left, left_pos)) =
                                    chunks[*c][*first_index].prev.map(|left_pos| {
                                        ((chunks[*c][left_pos].value, *pair1), (*c, left_pos))
                                    })
                                {
                                    if let Some(set) = decrements.get_mut(&left) {
                                        set.insert(left_pos);
                                    } else {
                                        decrements.insert(left, HashSet::from([left_pos]));
                                    }

                                    if let Some(set) = addons.get_mut(&(left.0, token)) {
                                        set.insert(left_pos);
                                    } else {
                                        addons.insert((left.0, token), HashSet::from([left_pos]));
                                    }
                                }

                                let current_pos = (*c, *first_index);

                                if let Some((right, right_pos)) =
                                    chunks[*c][second_index].next.map(|right_2pos| {
                                        ((*pair2, chunks[*c][right_2pos].value), (*c, second_index))
                                    })
                                {
                                    if let Some(set) = decrements.get_mut(&right) {
                                        set.insert(right_pos);
                                    } else {
                                        decrements.insert(right, HashSet::from([right_pos]));
                                    }

                                    if let Some(set) = addons.get_mut(&(token, right.1)) {
                                        set.insert(current_pos);
                                    } else {
                                        addons
                                            .insert((token, right.1), HashSet::from([current_pos]));
                                    }
                                }

                                chunks[*c][*first_index].next = chunks[*c][second_index].next;
                                chunks[*c][*first_index].value = token;
                                chunks[*c][second_index].next = None;
                            }
                        }
                        pq_counts.remove(&(*pair1, *pair2));

                        for (key, remove) in decrements {
                            pq_counts.change_priority_by(&key, |priority_set| {
                                for pos in remove {
                                    priority_set.0.remove(&pos);
                                }
                            });
                        }

                        for (key, insert) in addons {
                            pq_counts.push(key, OrdHashSet(insert));
                        }
                    }
                    None => break,
                }
            }
        }

        tokenizer
    }

    fn normalize(s: &str) -> String {
        let re = Regex::new(
            r"(?P<L>\p{L}+(?:-\p{L}+)*)|(?P<D>\p{N}+)|(?P<S>[▁ ]+)|(?P<O>[^\p{L}\p{N} ▁]+)",
        )
        .unwrap();

        let mut prev_type = GroupType::Other;

        re.replace_all(s, |caps: &Captures| {
            if let Some(m) = caps.name("L") {
                let res = if prev_type == GroupType::Other || prev_type == GroupType::Digits {
                    format!("▁{}", m.as_str())
                } else {
                    m.as_str().to_string()
                };
                prev_type = GroupType::Letters;
                res
            } else if let Some(m) = caps.name("D") {
                let res = if prev_type == GroupType::Other || prev_type == GroupType::Letters {
                    format!("▁{}", m.as_str())
                } else {
                    m.as_str().to_string()
                };
                prev_type = GroupType::Digits;
                res
            } else if let Some(m) = caps.name("O") {
                prev_type = GroupType::Other;
                m.as_str().to_string()
            } else if let Some(m) = caps.name("S") {
                let mut res = m.as_str().to_string();
                let end_idx = caps.get(0).unwrap().end();
                let remaining = &s[end_idx..];

                let next_is_letter = remaining.chars().next().is_some_and(|c| c.is_alphabetic());
                let next_is_digit = remaining.chars().next().is_some_and(|c| c.is_numeric());

                if !res.ends_with('\u{2581}') && (next_is_digit || next_is_letter) {
                    if (prev_type == GroupType::Letters && next_is_letter)
                        || (prev_type == GroupType::Digits && next_is_digit)
                    {
                        res.pop();
                    }
                    res.push('▁');
                }
                prev_type = GroupType::Spaces;
                res
            } else {
                "".to_string()
            }
        })
        .into_owned()
    }

    fn denormalize(s: &str) -> String {
        let s = Regex::new(r"(\p{L} *)▁(?=\p{L})|(\p{N} *)▁(?=\p{N})")
            .unwrap()
            .replace_all(s, "$1$2 ");
        Regex::new(r"(^|[^\p{L}] *)▁(?=\p{L})|(^|[^\p{N}] *)▁(?=\p{N})")
            .unwrap()
            .replace_all(&s, "$1$2")
            .to_string()
    }

    fn chunks(s: &str) -> Vec<&str> {
        let pattern =
            r"'s|'t|'re|'ve|'m|'ll|'d|▁?\p{L}+(?:-\p{L}+)*|▁?\p{N}{1,3}|[^\s\p{L}\p{N}▁]+|\s+";
        let re = Regex::new(pattern).unwrap();

        re.find_iter(s).map(|mat| mat.unwrap().as_str()).collect()
    }

    fn encode_normalized(&self, mut s: &str) -> Vec<Token> {
        let mut res = Vec::new();
        while !s.is_empty() {
            res.push(
                *(1..self.longest_str)
                    .rev()
                    .find_map(|l| {
                        if s.is_char_boundary(l) {
                            self.str2token.get(&s[..l]).inspect(|_| {
                                s = &s[l..];
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or(&self.str2token["[UNK]"]),
            );
        }
        res
    }

    pub fn encode(&self, s: &str) -> Vec<Token> {
        self.encode_normalized(&Tokenizer::normalize(s))
    }

    pub fn decode(&self, v: &[Token]) -> String {
        let start = std::time::Instant::now();
        let normalized = v
            .iter()
            .map(|&t| {
                self.token2str
                    .get(t as usize)
                    .map_or("[UNK]", |v| v)
                    .to_owned()
            })
            .collect::<Vec<_>>()
            .concat();
        println!(
            "decoded normalized text in {:?}",
            std::time::Instant::now() - start
        );
        Tokenizer::denormalize(&normalized)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::Instant};

    use super::*;
    const NORM_STRINGS: &[(&str, &str)] = &[
        ("ABC\nabc", "▁ABC\n▁abc"),
        ("ABC\n abc", "▁ABC\n ▁abc"),
        ("def ghi", "▁def▁ghi"),
        ("def  ghi", "▁def ▁ghi"),
        ("123 456", "▁123▁456"),
        ("123  456", "▁123 ▁456"),
        ("a-b", "▁a-b"),
        ("12bc", "▁12▁bc"),
        ("ABC\n123", "▁ABC\n▁123"),
        ("ABC\n 321", "▁ABC\n ▁321"),
        ("123 ghi", "▁123 ▁ghi"),
        ("def  123", "▁def  ▁123"),
        (" abcdef", " ▁abcdef"),
        (" 123", " ▁123"),
        ("===", "==="),
        ("a\nв парке", "▁a\n▁в▁парке"),
        ("a\nb cdef", "▁a\n▁b▁cdef"),
        ("2-nd", "▁2-▁nd"),
    ];

    #[test]
    fn norm() {
        for (source, target) in NORM_STRINGS {
            assert_eq!(Tokenizer::normalize(source), *target)
        }
    }

    #[test]
    fn twice_norm() {
        for (source, target) in [
            ("▁a", "▁a"),
            ("▁1", "▁1"),
            ("▁def  ▁123", "▁def  ▁123"),
            ("▁a\n▁b", "▁a\n▁b"),
            ("▁a\n ▁b", "▁a\n ▁b"),
            ("▁a\n▁в▁парке", "▁a\n▁в▁парке"),
        ] {
            assert_eq!(Tokenizer::normalize(source), *target)
        }
    }

    #[test]
    fn denorm() {
        for (target, source) in NORM_STRINGS {
            assert_eq!(Tokenizer::denormalize(source), *target)
        }
    }

    #[test]
    fn twice_denorm() {
        for (source, target) in [("a a", "a a"), ("a  a", "a  a"), ("def  123", "def  123")] {
            assert_eq!(Tokenizer::denormalize(source), *target)
        }
    }

    const STRINGS: &[&str] = &[
        "кто-то где-то что-то с чем-то смешивает- смешивает да ка-а-ак смешает",
        "пробелы    пробелы[UNK]---=== (((Hello, World!))) ===---\n\
        1000000 / 1000 = 1000",
        "Кружка-термос на 0.5л (вмещает 50/64 см³, вес 516г).\
        Скорость составила 90км/ч, а длина кабеля — 15мм. 
        В коробке 24шт. товара по цене 150руб/шт.\
        Это произошло в 90-х годах XX века. На 2-м этаже открылся новый офис. 
        В 10-12 часах езды от города находится заповедник. 
        Выпускники 11-го класса сдали экзамены на 95-100 баллов\
        Модель процессора: Intel Core i7-12700K или Эльбрус-8С.",
    ];

    #[test]
    fn codepoint2token() {
        let tok = Tokenizer::train(&STRINGS.concat(), None);
        for &s in STRINGS.iter() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
        }
    }

    #[test]
    fn codepoint_bpe() {
        let tok = Tokenizer::train(&STRINGS.concat(), Some(512));
        for &s in STRINGS.iter() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
            println!(
                "original byte len: {}, encoded byte len: {}",
                s.len(),
                enc.len() * 2
            )
        }
    }

    #[test]
    fn shakespeare() {
        let s = fs::read_to_string("shakespeare.txt").unwrap();
        let start = Instant::now();
        let tok = Tokenizer::train(&s, Some(512));
        println!("trained in {:?}", Instant::now() - start);
        let start = Instant::now();
        let enc = tok.encode(&s);
        println!("encoded in {:?}", Instant::now() - start);
        let start = Instant::now();
        let dec = tok.decode(&enc);
        println!("decoded in {:?}", Instant::now() - start);
        assert_eq!(s, dec);
        println!(
            "original byte len: {}, encoded byte len: {}",
            s.len(),
            enc.len() * 2
        )
    }
}
