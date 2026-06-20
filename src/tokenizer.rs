mod arena_list;
mod pair_ordering;
mod utils;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use priority_queue::PriorityQueue;
use rustc_hash::FxHashMap;

use crate::{
    Token,
    prepare::{chunks, denormalize, normalize},
    tokenizer::{
        arena_list::ArenaList,
        pair_ordering::PairOrd,
        utils::{Export, Import, ProtoToken},
    },
};

static BYTES: [u8; 256] = {
    let mut arr = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        arr[i] = i as u8;
        i += 1;
    }
    arr
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
            let mut pair_positions = FxHashMap::default();
            for (piece_id, (chunk_v, _scale)) in pieces.iter().enumerate() {
                for (index, p) in chunk_v.raw_pairs().enumerate() {
                    pair_positions
                        .entry((p[0].value, p[1].value))
                        .or_insert(Vec::new())
                        .push((piece_id, index));
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

            let mut decrements: FxHashMap<(u16, u16), u32> = FxHashMap::default();
            let mut addons: FxHashMap<(u16, u16), Vec<(usize, usize)>> = FxHashMap::default();
            while protostack.len() < vocab_size {
                match pq_counts.pop() {
                    Some((pair, _)) => {
                        let token = protostack.len() as Token;
                        protostack.push(ProtoToken::Pair(pair.0 as usize, pair.1 as usize));

                        let positions: Vec<_> = pair_positions.remove(&pair).unwrap();
                        for (piece_id, ab_pos) in positions {
                            // XABY -> XTY: [XA]--, [BY]--, AB=>T, [XT]++, [TY]++
                            if Some(pair) == pieces[piece_id].0.pair_at(ab_pos) {
                                if let Some((xa, _xa_pos)) =
                                    pieces[piece_id].0.prev_pair_pos(ab_pos)
                                {
                                    if let Some(count) = decrements.get_mut(&xa) {
                                        *count += pieces[piece_id].1;
                                    } else {
                                        decrements.insert(xa, pieces[piece_id].1);
                                    }
                                }
                                if let Some((by, _by_pos)) =
                                    pieces[piece_id].0.next_pair_pos(ab_pos)
                                {
                                    if let Some(count) = decrements.get_mut(&by) {
                                        *count += pieces[piece_id].1;
                                    } else {
                                        decrements.insert(by, pieces[piece_id].1);
                                    }
                                }
                                let (xt_opt, ty_opt) = pieces[piece_id].0.fuse_into(ab_pos, token);
                                if let Some((xt, xt_pos)) = xt_opt {
                                    if let Some(set) = addons.get_mut(&xt) {
                                        set.push((piece_id, xt_pos));
                                    } else {
                                        addons.insert(xt, Vec::from([(piece_id, xt_pos)]));
                                    }
                                }
                                if let Some((ty, ty_pos)) = ty_opt {
                                    if let Some(set) = addons.get_mut(&ty) {
                                        set.push((piece_id, ty_pos));
                                    } else {
                                        addons.insert(ty, Vec::from([(piece_id, ty_pos)]));
                                    }
                                }
                            }
                        }

                        for (key, decrement) in decrements.drain() {
                            pq_counts.change_priority_by(&key, |pair_priority| {
                                pair_priority.0 -= decrement
                            });
                        }

                        for (key, insert) in addons.drain() {
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
                tokenizer
                    .token2str
                    .push(pt.pieces(i as Token, &tokenizer.token2str, &protostack));
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
