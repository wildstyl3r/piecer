use crate::utils::{ArenaNode, ProtoToken};
use std::collections::{HashMap, HashSet};

mod pair_ordering;
mod prepare;
mod utils;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use priority_queue::PriorityQueue;

use crate::{
    pair_ordering::StableOrdHashSet,
    prepare::{chunks, denormalize, normalize},
};

pub type Token = u16;

pub struct Tokenizer {
    token2str: Vec<String>,
    str2token_ac: DoubleArrayAhoCorasick<Token>,
}

impl Tokenizer {
    pub fn train(s: &str, vocab_size: Option<usize>) -> Self {
        let mut alphabet: Vec<_> = (0..0x20u8)
            .map(|b| {
                if b == b'\t' || b == b'\n' || b == b'\r' {
                    (b as char).to_string()
                } else {
                    format!("<{:02x}>", b)
                }
            })
            .chain((0x20u8..0x7f).map(|b| (b as char).to_string()))
            .chain((0x7f..=0xffu8).map(|b| format!("<{:02x}>", b)))
            .collect();
        let mut extension = s
            .chars()
            .filter(|c| c.len_utf8() > 1)
            .fold(
                HashSet::<String>::from(
                    ["▁", "[UNK]", "[PAD]", "[BOS]", "[EOS]"].map(|s| s.to_string()),
                ),
                |mut acc, c| {
                    acc.insert(c.to_string());
                    acc
                },
            )
            .into_iter()
            .collect::<Vec<_>>();
        extension.sort();
        alphabet.extend(extension);

        static BYTES: [u8; 256] = {
            let mut arr = [0u8; 256];
            let mut i = 0;
            while i < 256 {
                arr[i] = i as u8;
                i += 1;
            }
            arr
        };

        let mut tokenizer = Self {
            str2token_ac: DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(
                    (0..256)
                        .map(|i| &BYTES[i..=i])
                        .chain(alphabet[256..].iter().map(|tok| tok.as_bytes())),
                )
                .unwrap(),
            token2str: alphabet,
        };
        let mut protostack: Vec<_> = (0..tokenizer.token2str.len())
            .map(|i| ProtoToken::Token(i as Token))
            .collect();

        if let Some(vocab_size) = vocab_size {
            let mut str2id = HashMap::new();
            let mut pieces: Vec<(Vec<ArenaNode>, u32)> =
                chunks(&normalize(s)).fold(Vec::new(), |mut pieces, str_chunk| {
                    if let Some(id) = str2id.get(str_chunk) {
                        pieces[*id as usize].1 += 1;
                        pieces
                    } else {
                        let id = str2id.len() as u32;
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
                        pieces.push((chunk, id));
                        str2id.insert(str_chunk, id);
                        pieces
                    }
                });
            let mut bootstrap_counts = HashMap::new();
            for (id, (chunk_v, scale)) in pieces.iter().enumerate() {
                for (i, p) in chunk_v.windows(2).enumerate() {
                    bootstrap_counts
                        .entry((p[0].value, p[1].value))
                        .or_insert((Vec::new(), scale))
                        .0
                        .push((id, i));
                }
            }
            let mut pq_counts = PriorityQueue::with_capacity(bootstrap_counts.len());
            for (k, v) in bootstrap_counts {
                pq_counts.push(k, StableOrdHashSet(v.0.into_iter().collect(), k, *v.1));
            }
            while protostack.len() < vocab_size {
                match pq_counts.pop() {
                    Some(((pair1, pair2), StableOrdHashSet(positions_set, _, scale))) => {
                        let token = protostack.len() as Token;
                        protostack.push(ProtoToken::Pair(pair1 as usize, pair2 as usize));

                        let mut positions: Vec<_> = positions_set.into_iter().collect();
                        positions.sort_by_key(|(_chunk_id, i)| *i);
                        let mut decrements: HashMap<(u16, u16), Vec<(usize, usize)>> =
                            HashMap::new();
                        let mut addons: HashMap<(u16, u16), HashSet<(usize, usize)>> =
                            HashMap::new();
                        for (id, first_index) in positions.into_iter().rev() {
                            if let Some(second_index) = pieces[id].0[first_index].next {
                                if let Some((left, left_pos)) =
                                    pieces[id].0[first_index].prev.map(|left_pos| {
                                        ((pieces[id].0[left_pos].value, pair1), (id, left_pos))
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

                                let current_pos = (id, first_index);

                                if let Some((right, right_pos)) =
                                    pieces[id].0[second_index].next.map(|right_2pos| {
                                        (
                                            (pair2, pieces[id].0[right_2pos].value),
                                            (id, second_index),
                                        )
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

                                pieces[id].0[first_index].next = pieces[id].0[second_index].next;
                                pieces[id].0[first_index].value = token;
                                pieces[id].0[second_index].next = None;
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
                            pq_counts.push(key, StableOrdHashSet(insert, key, scale));
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
            }
            tokenizer.str2token_ac = DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(
                    (0..256)
                        .map(|i| &BYTES[i..=i])
                        .chain(tokenizer.token2str[256..].iter().map(|tok| tok.as_bytes())),
                )
                .unwrap();
        }
        tokenizer
    }

    fn encode_normalized(&self, s: &str) -> Vec<Token> {
        self.str2token_ac
            .leftmost_find_iter(s)
            .map(|mat| mat.value())
            .collect()
    }

    pub fn encode(&self, s: &str) -> Vec<Token> {
        self.encode_normalized(&normalize(s))
    }

    fn decode_normalized(&self, v: &[Token]) -> String {
        let mut result = String::new();
        let mut byte_buff = Vec::new();
        for &t in v {
            if t == 0x09 || t == 0x0a || t == 0x0d || (0x20..0x7f).contains(&t) || t > 0xff {
                for &b in &byte_buff {
                    result.push_str(self.token2str[b as usize].as_str());
                }
                byte_buff.clear();
                result.push_str(self.token2str[t as usize].as_str())
            } else {
                if (0xc2..=0xf4).contains(&t) {
                    for &b in &byte_buff {
                        result.push_str(self.token2str[b as usize].as_str());
                    }
                    byte_buff.clear();
                }
                byte_buff.push(t as u8);
                if let Ok(s) = str::from_utf8(&byte_buff) {
                    result.push_str(s);
                    byte_buff.clear();
                }
            }
        }
        for &b in &byte_buff {
            result.push_str(self.token2str[b as usize].as_str());
        }
        result
    }

    pub fn decode(&self, v: &[Token]) -> String {
        denormalize(&self.decode_normalized(v))
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

    #[test]
    fn byte_fallback() {
        let tok = Tokenizer::train(&STRINGS.concat(), None);
        let s = "龍";
        // assert_eq!("<e9><be><8d>", tok.decode(&tok.encode(s)));
        assert_eq!(s, tok.decode(&tok.encode(s)));
    }
}
