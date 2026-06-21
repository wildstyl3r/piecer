mod arena_list;
mod pair_ordering;
mod utils;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    mem::size_of,
    path::Path,
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use priority_queue::PriorityQueue;
use rustc_hash::FxHashMap;

use crate::{
    Token, TokenType,
    error::TokenizerError,
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

/// Construct via [`Tokenizer::train`] or [`Tokenizer::load`]. Use by calling [`encode`](Tokenizer::encode)
/// / [`decode`](Tokenizer::decode) or inspect the vocabulary with
/// [`Tokenizer::vocab_size`], [`Tokenizer::token_to_string`], [`Tokenizer::str_to_token`], and [`Tokenizer::tokens`].
///
/// # Token representation
///
/// Tokens are `T` values where `T: `[`TokenType`]. By default, `T = `[`Token`](`u16`).
/// Tokens `0..256` correspond to single bytes (with printable
/// ASCII mapped to their character representation and control/high bytes to `<hex>` notation). 257th token is the word-start symbol `▁`.
/// Tokens `257..` are special tokens, plus Unicode codepoints extracted from the training data, plus those learned by BPE merges (if any).
#[non_exhaustive]
pub struct Tokenizer<T: TokenType = Token> {
    token2str: Vec<String>,
    str2token_ac: DoubleArrayAhoCorasick<T>,
}

impl<T: TokenType> Tokenizer<T> {
    /// Returns the total vocabulary size (byte tokens + codepoint + BPE merge tokens).
    pub fn vocab_size(&self) -> usize {
        self.token2str.len()
    }

    /// Maps a token ID to its string representation.
    ///
    /// Returns `None` if the token ID exceeds the vocabulary range.
    /// For byte tokens `0..256`, this returns the byte's display form
    /// (printable ASCII as-is, others as `<hex>`).
    pub fn token_to_string(&self, tok: T) -> Option<&str> {
        self.token2str.get(tok.to_index()).map(|s| s.as_str())
    }

    /// Maps a string to its token ID, if the string corresponds to a single token.
    ///
    /// Returns `None` if the string is not present as an atomic vocabulary entry.
    /// Note that multi-token sequences (e.g. `"hello world"`) will not match —
    /// this only succeeds when the entire input maps to exactly one token.
    pub fn str_to_token(&self, s: &str) -> Option<T> {
        let mut it = self.str2token_ac.leftmost_find_iter(s).map(|m| m.value());
        match (it.next(), it.next()) {
            (Some(tok), None) => {
                if self.token2str[tok.to_index()] == s {
                    Some(tok)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Iterates over all `(token_id, string)` pairs in the vocabulary.
    ///
    /// Byte tokens come first (0..256), followed by sorted Unicode codepoints
    /// and BPE merge tokens in merge order.
    pub fn tokens(&self) -> impl Iterator<Item = (T, &str)> {
        self.token2str
            .iter()
            .enumerate()
            .map(|(i, s)| (T::from_index(i), s.as_str()))
    }

    /// Loads a tokenizer from a JSON file previously saved with [`Tokenizer::save`].
    pub fn load(path: &Path) -> Result<Self, TokenizerError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let external: Import = serde_json::from_reader(reader)?;

        if external
            .version
            .split(".")
            .next()
            .map(|major| major == env!("CARGO_PKG_VERSION_MAJOR"))
            .unwrap_or(false)
        {
            let vocab_size = 257 + external.vocabulary.len();
            if vocab_size > T::max_index() + 1 {
                return Err(TokenizerError::Vocabulary(format!(
                    "vocabulary size {} exceeds maximum for token type (max {})",
                    vocab_size,
                    T::max_index() + 1
                )));
            }
            Ok(Self {
                str2token_ac: DoubleArrayAhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(
                        (0..256)
                            .map(|i| &BYTES[i..=i])
                            .chain(std::iter::once("▁".as_bytes()))
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
                    .chain(std::iter::once("▁".to_string()))
                    .chain(external.vocabulary)
                    .collect::<Vec<_>>(),
            })
        } else {
            Err(TokenizerError::Vocabulary(external.version))
        }
    }

    /// Saves the tokenizer vocabulary to a JSON file.
    ///
    /// Only the learned vocabulary (257..) is stored; the first 256 byte-tokens plus word-start `▁`
    /// are implicit. The file also records the crate version for forward compatibility.
    pub fn save(&self, path: &Path) -> Result<(), TokenizerError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(
            writer,
            &Export {
                version: env!("CARGO_PKG_VERSION"),
                vocabulary: &self.token2str[257..],
            },
        )?;
        Ok(())
    }

    /// Trains a new BPE tokenizer on the given text.
    ///
    /// The initial vocabulary always contains 257 initial tokens (256 single bytes plus the word start symbol `▁`),
    /// in addition to the provided special tokens.
    ///
    /// In case `max_extra_tokens` is `Some(n)`, other unicode codepoints found in `s` are appended
    /// to the vocabulary first, and then BPE merge operations are applied until the vocabulary
    /// reaches `257 + special_tokens.len() + n` entries.
    ///
    /// If n is less than the number of unicode codepoints, then only the top-n are appended
    /// from those in `s`, sorted longest-first, then lexicographically.
    pub fn train(s: &str, special_tokens: &[&str], max_extra_tokens: Option<usize>) -> Self {
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
            .chain(std::iter::once("▁".to_string()))
            .collect();
        let mut extension = s
            .chars()
            .filter(|c| c.len_utf8() > 1)
            .fold(HashSet::<String>::new(), |mut acc, c| {
                acc.insert(c.to_string());
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();
        extension.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        alphabet.extend(
            special_tokens.iter().map(|&s| s.to_string()).chain(
                extension
                    .into_iter()
                    .take(max_extra_tokens.unwrap_or(usize::MAX)),
            ),
        );

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
            .map(|i| ProtoToken::Token(T::from_index(i)))
            .collect();

        if let Some(extra_tokens) = max_extra_tokens {
            let mut str2id = HashMap::new();
            let mut pieces: Vec<(ArenaList<T>, u32)> =
                chunks(&normalize(s))
                    .into_iter()
                    .fold(Vec::new(), |mut pieces, str_chunk| {
                        if let Some(id) = str2id.get(str_chunk) {
                            pieces[*id as usize].1 += 1;
                            pieces
                        } else {
                            let id = str2id.len() as u32;
                            let piece: ArenaList<T> =
                                tokenizer.encode_normalized(str_chunk).collect();
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

            let mut decrements: FxHashMap<(T, T), u32> = FxHashMap::default();
            let mut addons: FxHashMap<(T, T), Vec<(usize, usize)>> = FxHashMap::default();
            while protostack.len() - 257 < extra_tokens {
                if protostack.len() > T::max_index() {
                    break;
                }
                match pq_counts.pop() {
                    Some((pair, _)) => {
                        let token = T::from_index(protostack.len());
                        protostack.push(ProtoToken::Pair(pair.0.to_index(), pair.1.to_index()));

                        let positions: Vec<_> = pair_positions.remove(&pair).unwrap();
                        for (piece_id, ab_pos) in positions {
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
                tokenizer.token2str.push(pt.pieces(
                    T::from_index(i),
                    &tokenizer.token2str,
                    &protostack,
                ));
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

    fn encode_normalized(&self, s: &str) -> impl Iterator<Item = T> {
        self.str2token_ac
            .leftmost_find_iter(s)
            .map(|mat| mat.value())
    }

    /// Encodes a string into a sequence of token IDs.
    ///
    /// The input is first normalized, then chunked by a GPT-style regex pattern,
    /// and each chunk is matched against the vocabulary using leftmost-longest
    /// Aho-Corasick search. Characters not in the vocabulary fall back to their
    /// UTF-8 byte tokens.
    pub fn encode(&self, s: &str) -> Vec<T> {
        self.encode_normalized(&normalize(s)).collect()
    }

    fn decode_normalized(&self, v: &[T]) -> String {
        let mut result = String::new();
        let mut byte_buff = Vec::new();
        for &t in v {
            let t_u32 = t.to_u32();
            if t_u32 == 0x09
                || t_u32 == 0x0a
                || t_u32 == 0x0d
                || (0x20u32..0x7fu32).contains(&t_u32)
                || t_u32 > 0xff
            {
                for &b in &byte_buff {
                    result.push_str(self.token2str[b as usize].as_str());
                }
                byte_buff.clear();
                result.push_str(self.token2str[t.to_index()].as_str())
            } else {
                if (0xc2u32..=0xf4u32).contains(&t_u32) {
                    for &b in &byte_buff {
                        result.push_str(self.token2str[b as usize].as_str());
                    }
                    byte_buff.clear();
                }
                byte_buff.push(t_u32 as u8);
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

    /// Decodes a sequence of token IDs back into a string.
    ///
    /// Consecutive byte tokens that form valid UTF-8 are reassembled into
    /// characters. The result is then denormalized (removing or converting `▁` markers
    /// back to spaces where appropriate).
    pub fn decode(&self, v: &[T]) -> String {
        denormalize(&self.decode_normalized(v))
    }

    /// Returns the size in bytes of a single token value.
    pub fn token_size() -> usize {
        size_of::<T>()
    }
}
