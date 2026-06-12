use crate::utils::{ArenaNode, ProtoToken};
use std::collections::{HashMap, HashSet};

mod pair_ordering;
mod prepare;
mod utils;

use aho_corasick::AhoCorasick;
use priority_queue::PriorityQueue;

use crate::{
    pair_ordering::StableOrdHashSet,
    prepare::{chunks, denormalize, normalize},
};

pub type Token = u16;

pub struct Tokenizer {
    str2token: HashMap<String, Token>,
    token2str: Vec<String>,
    str2token_ac: AhoCorasick,
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
            str2token_ac: AhoCorasick::builder()
                .kind(Some(aho_corasick::AhoCorasickKind::ContiguousNFA))
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(&alphabet)
                .unwrap(),
            token2str: alphabet,
        };
        let mut protostack: Vec<_> = (0..tokenizer.token2str.len())
            .map(|i| ProtoToken::Token(i as Token))
            .collect();

        if let Some(vocab_size) = vocab_size {
            let norm = normalize(s);
            let start = std::time::Instant::now();
            let mut chunks: Vec<Vec<ArenaNode>> = chunks(&norm)
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
            println!("chunking done in {:?}", std::time::Instant::now() - start);

            let start = std::time::Instant::now();
            let mut bootstrap_counts = HashMap::new();
            for (chunk, chunk_v) in chunks.iter().enumerate() {
                for (i, p) in chunk_v.windows(2).enumerate() {
                    bootstrap_counts
                        .entry((p[0].value, p[1].value))
                        .or_insert(Vec::new())
                        .push((chunk, i));
                }
            }
            println!(
                "count construction done in {:?}",
                std::time::Instant::now() - start
            );
            let mut pq_counts = PriorityQueue::with_capacity(bootstrap_counts.len());
            for (k, v) in bootstrap_counts {
                pq_counts.push(k, StableOrdHashSet(v.into_iter().collect(), k));
            }
            while protostack.len() < vocab_size {
                match pq_counts.pop() {
                    Some(((pair1, pair2), positions_set)) => {
                        let token = protostack.len() as Token;
                        protostack.push(ProtoToken::Pair(pair1 as usize, pair2 as usize));

                        let mut positions: Vec<_> = positions_set.0.into_iter().collect();
                        positions.sort_by_key(|(_chunk, i)| *i);
                        let mut decrements: HashMap<(u16, u16), Vec<(usize, usize)>> =
                            HashMap::new();
                        let mut addons: HashMap<(u16, u16), HashSet<(usize, usize)>> =
                            HashMap::new();
                        for (c, first_index) in positions.into_iter().rev() {
                            if let Some(second_index) = chunks[c][first_index].next {
                                if let Some((left, left_pos)) =
                                    chunks[c][first_index].prev.map(|left_pos| {
                                        ((chunks[c][left_pos].value, pair1), (c, left_pos))
                                    })
                                {
                                    if let Some(set) = decrements.get_mut(&left) {
                                        set.push(left_pos);
                                    } else {
                                        decrements.insert(left, vec![left_pos]);
                                    }

                                    if let Some(set) = addons.get_mut(&(left.0, token)) {
                                        set.insert(left_pos);
                                    } else {
                                        addons.insert((left.0, token), HashSet::from([left_pos]));
                                    }
                                }

                                let current_pos = (c, first_index);

                                if let Some((right, right_pos)) =
                                    chunks[c][second_index].next.map(|right_2pos| {
                                        ((pair2, chunks[c][right_2pos].value), (c, second_index))
                                    })
                                {
                                    if let Some(set) = decrements.get_mut(&right) {
                                        set.push(right_pos);
                                    } else {
                                        decrements.insert(right, vec![right_pos]);
                                    }

                                    if let Some(set) = addons.get_mut(&(token, right.1)) {
                                        set.insert(current_pos);
                                    } else {
                                        addons
                                            .insert((token, right.1), HashSet::from([current_pos]));
                                    }
                                }

                                chunks[c][first_index].next = chunks[c][second_index].next;
                                chunks[c][first_index].value = token;
                                chunks[c][second_index].next = None;
                            }
                        }

                        for (key, remove) in decrements {
                            pq_counts.change_priority_by(&key, |priority_set| {
                                for pos in remove {
                                    priority_set.0.remove(&pos);
                                }
                            });
                        }

                        for (key, insert) in addons {
                            pq_counts.push(key, StableOrdHashSet(insert, key));
                        }
                    }
                    None => break,
                }
            }
            let base_len = tokenizer.token2str.len();
            for (i, pt) in protostack.iter().enumerate().skip(base_len) {
                let s = pt
                    .pieces(i as Token, &tokenizer.token2str, &protostack)
                    .concat();
                tokenizer.token2str.push(s.clone());
                tokenizer.str2token.insert(s, i as Token);
            }
            tokenizer.str2token_ac = AhoCorasick::builder()
                .kind(Some(aho_corasick::AhoCorasickKind::ContiguousNFA))
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(&tokenizer.token2str)
                .unwrap();
        }
        tokenizer
    }

    fn encode_normalized(&self, s: &str) -> Vec<Token> {
        self.str2token_ac
            .find_iter(s)
            .map(|mat| mat.pattern().as_u32() as Token)
            .collect()
    }

    pub fn encode(&self, s: &str) -> Vec<Token> {
        self.encode_normalized(&normalize(s))
    }

    pub fn decode(&self, v: &[Token]) -> String {
        let start = std::time::Instant::now();
        let normalized: String = v
            .iter()
            .map(|&t| self.token2str.get(t as usize).map_or("[UNK]", |v| v))
            .collect();
        println!(
            "decoded normalized text in {:?}",
            std::time::Instant::now() - start
        );
        denormalize(&normalized)
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
            assert_eq!(normalize(source), *target)
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
            assert_eq!(normalize(source), *target)
        }
    }

    #[test]
    fn denorm() {
        for (target, source) in NORM_STRINGS {
            assert_eq!(denormalize(source), *target)
        }
    }

    #[test]
    fn twice_denorm() {
        for (source, target) in [("a a", "a a"), ("a  a", "a  a"), ("def  123", "def  123")] {
            assert_eq!(denormalize(source), *target)
        }
    }

    const STRINGS: &[&str] = &[
        "кто-то где-то что-то с чем-то смешивает- смешивает да ка-а-ак смешает",
        "пробелы    пробелы[UNK]---=== (((Hello, World!))) ===---\n\
        1000000 / 1000 = 1000",
        "Кружка-термос на 0.5л (вмещает 50/64 см³, вес 516г).\
        Скорость составила 90км/ч, а длина кабеля — 15мм.",
        "В коробке 24шт. товара по цене 150руб/шт.\
        Это произошло в 90-х годах XX века. На 2-м этаже открылся новый офис.
        В 10-12 часах езды от города находится заповедник.
        Выпускники 11-го класса сдали экзамены на 95-100 баллов",
        "Модель процессора: Intel Core i7-12700K или Эльбрус-8С.",
    ];

    #[test]
    fn chunking() {
        let pattern =
            r"'s|'t|'re|'ve|'m|'ll|'d|▁?\p{L}+(?:-\p{L}+)*|▁?\p{N}{1,3}|[^\p{L}\p{N}▁\s]+|\s+";
        let re = regex::Regex::new(pattern).unwrap();

        for &s in STRINGS.iter() {
            let ns = normalize(s);
            let chunks = chunks(&ns);
            let regex_chunks = re.find_iter(&ns).map(|mat| mat.as_str());
            assert!(chunks.eq(regex_chunks));
        }

        let ss = fs::read_to_string("shakespeare.txt").unwrap();
        let ns = normalize(&ss);
        let chunks = chunks(&ns);
        let regex_chunks = re.find_iter(&ns).map(|mat| mat.as_str());
        assert!(chunks.eq(regex_chunks));
    }

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
        let s0 = Instant::now();
        let start = Instant::now();
        let tok = Tokenizer::train(&s, Some(10000));
        println!("trained in {:?}", Instant::now() - start);
        let start = Instant::now();
        let enc = tok.encode(&s);
        println!("encoded in {:?}", Instant::now() - start);
        let start = Instant::now();
        let dec = tok.decode(&enc);
        println!("decoded in {:?}", Instant::now() - start);
        assert_eq!(s, dec);
        println!("total time {:?}", Instant::now() - s0);
        println!(
            "original byte len: {}, encoded byte len: {}",
            s.len(),
            enc.len() * 2
        )
    }
}
