mod arena_list;
mod pair_ordering;
mod utils;

use crate::{
    Token,
    tokenizer::{
        arena_list::ArenaList,
        utils::{Export, Import, ProtoToken},
    },
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use priority_queue::PriorityQueue;

use crate::{
    prepare::{chunks, denormalize, normalize},
    tokenizer::pair_ordering::PairOrd,
};

pub struct Tokenizer {
    token2str: Vec<String>,
    str2token_ac: DoubleArrayAhoCorasick<Token>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let external: Import = serde_json::from_reader(reader).map_err(std::io::Error::other)?;
        static BYTES: [u8; 256] = {
            let mut arr = [0u8; 256];
            let mut i = 0;
            while i < 256 {
                arr[i] = i as u8;
                i += 1;
            }
            arr
        };
        Ok(Self {
            str2token_ac: DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(
                    (0..256)
                        .map(|i| &BYTES[i..=i])
                        .chain(external.vocabulary.iter().map(|tok| tok.as_bytes())),
                )
                .unwrap(),
            token2str: (0..0x20u8)
                .map(|b| {
                    if b == b'\t' || b == b'\n' || b == b'\r' {
                        (b as char).to_string()
                    } else {
                        format!("<{:02x}>", b)
                    }
                })
                .chain((0x20u8..0x7f).map(|b| (b as char).to_string()))
                .chain((0x7f..=0xffu8).map(|b| format!("<{:02x}>", b)))
                .chain(external.vocabulary)
                .collect::<Vec<_>>(),
        })
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(
            writer,
            &Export {
                version: env!("CARGO_PKG_VERSION"),
                vocabulary: &self.token2str[256..],
            },
        )
        .map_err(std::io::Error::other)?;
        Ok(())
    }

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
            let mut pieces: Vec<(ArenaList, u32)> =
                chunks(&normalize(s))
                    .into_iter()
                    .fold(Vec::new(), |mut pieces, str_chunk| {
                        if let Some(id) = str2id.get(str_chunk) {
                            pieces[*id as usize].1 += 1;
                            pieces
                        } else {
                            let id = str2id.len() as u32;
                            let piece: ArenaList =
                                tokenizer.encode_normalized(str_chunk).iter().collect();
                            pieces.push((piece, 1));
                            str2id.insert(str_chunk, id);
                            pieces
                        }
                    });
            let mut pair_positions = HashMap::new();
            for (piece_id, (chunk_v, _scale)) in pieces.iter().enumerate() {
                for (index, p) in chunk_v.raw_pairs().enumerate() {
                    pair_positions
                        .entry((p[0].value, p[1].value))
                        .or_insert(HashSet::new())
                        .insert((piece_id, index));
                }
            }
            let mut pq_counts = PriorityQueue::with_capacity(pair_positions.len());
            for (k, v) in &pair_positions {
                pq_counts.push(
                    *k,
                    PairOrd(
                        v.iter()
                            .fold(0, |acc, (piece_id, _index)| acc + pieces[*piece_id].1),
                        *k,
                    ),
                );
            }
            while protostack.len() < vocab_size {
                match pq_counts.pop() {
                    Some((pair, _)) => {
                        let token = protostack.len() as Token;
                        protostack.push(ProtoToken::Pair(pair.0 as usize, pair.1 as usize));

                        let mut positions: Vec<_> =
                            pair_positions.remove(&pair).unwrap().into_iter().collect();
                        positions.sort_by_key(|(_, i)| *i);
                        let mut decrements: HashMap<(u16, u16), Vec<(usize, usize)>> =
                            HashMap::new();
                        let mut addons: HashMap<(u16, u16), HashSet<(usize, usize)>> =
                            HashMap::new();
                        for (piece_id, ab_pos) in positions {
                            // XABY -> XTY: [XA]--, [BY]--, AB=>T, [XT]++, [TY]++
                            if pieces[piece_id].0.pair_at(ab_pos).is_some() {
                                if let Some((xa, xa_pos)) = pieces[piece_id].0.prev_pair_pos(ab_pos)
                                {
                                    if let Some(set) = decrements.get_mut(&xa) {
                                        set.push((piece_id, xa_pos));
                                    } else {
                                        decrements.insert(xa, vec![(piece_id, xa_pos)]);
                                    }
                                }
                                if let Some((by, by_pos)) = pieces[piece_id].0.next_pair_pos(ab_pos)
                                {
                                    if let Some(set) = decrements.get_mut(&by) {
                                        set.push((piece_id, by_pos));
                                    } else {
                                        decrements.insert(by, vec![(piece_id, by_pos)]);
                                    }
                                }
                                let (xt_opt, ty_opt) = pieces[piece_id].0.fuse_into(ab_pos, token);
                                if let Some((xt, xt_pos)) = xt_opt {
                                    if let Some(set) = addons.get_mut(&xt) {
                                        set.insert((piece_id, xt_pos));
                                    } else {
                                        addons.insert(xt, HashSet::from([(piece_id, xt_pos)]));
                                    }
                                }
                                if let Some((ty, ty_pos)) = ty_opt {
                                    if let Some(set) = addons.get_mut(&ty) {
                                        set.insert((piece_id, ty_pos));
                                    } else {
                                        addons.insert(ty, HashSet::from([(piece_id, ty_pos)]));
                                    }
                                }
                            }
                        }

                        for (key, remove) in decrements {
                            if let Some(positions) = pair_positions.get_mut(&key) {
                                for pos in &remove {
                                    positions.remove(pos);
                                }
                            }
                            pq_counts.change_priority_by(&key, |pair_priority| {
                                for (piece_id, _index) in remove {
                                    pair_priority.0 -= pieces[piece_id].1;
                                }
                            });
                        }

                        for (key, insert) in addons {
                            pq_counts.push(
                                key,
                                PairOrd(
                                    insert.iter().fold(0, |acc, (piece_id, _index)| {
                                        acc + pieces[*piece_id].1
                                    }),
                                    key,
                                ),
                            );
                            pair_positions.insert(key, insert);
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
