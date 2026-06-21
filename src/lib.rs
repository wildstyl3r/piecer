/*!
A tokenizer operating on Unicode codepoints. Supports automatic byte-fallback
for out-of-vocabulary characters and with optional [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding)
merging mode.

The core vocabulary always includes the full byte range (`0x00`–`0xff`), special token `▁` plus
any multi-byte Unicode codepoints found in the training data. During encoding,
any input that cannot be mapped to multi-byte vocabulary elements decomposes
into a sequence of single-byte tokens. On decoding, consecutive byte tokens
that form valid UTF-8 are reassembled into characters; invalid sequences
render as `<hex>` notation (e.g. `<e9><be><8d>`).

Unicode letters, digits, whitespace characters and "any symbols not belonging to these types"
are guaranteed to never be mixed during a token merge. Words and standalone numbers are prepended by a
special symbol `▁` internally to indicate a word or a number start. Numbers are split into parts consisting
of no more than 3 digits. and optionally `▁`, if it is in the beginning of the number.

# Quick start

```
use piecer::Tokenizer;

// Train a tokenizer with up to 512 merge operations
let tok: Tokenizer = Tokenizer::train("hello world", &["[PAD]"], Some(512));

let tokens = tok.encode("hello world");
let decoded = tok.decode(&tokens);
assert_eq!("hello world", decoded);
```

# Byte fallback

Characters absent from the training vocabulary are encoded as their UTF-8
byte tokens and transparently reassembled on decode:

```
use piecer::Tokenizer;

let tok: Tokenizer = Tokenizer::train("hello", &[], Some(512));
let tokens = tok.encode("龍");  // U+9F8D — not in training data
assert_eq!("龍", tok.decode(&tokens));  // reassembled from <e9><be><8d>
 ```

# Persistence

```
use piecer::Tokenizer;
use std::path::Path;

let tok: Tokenizer = Tokenizer::train("some training text", &[], Some(512));
tok.save(Path::new("my_tokenizer.json")).unwrap();
let loaded: Tokenizer = Tokenizer::load(Path::new("my_tokenizer.json")).unwrap();
assert_eq!(tok.vocab_size(), loaded.vocab_size());
```

 */

mod error;
mod prepare;
pub mod tokenizer;

pub use crate::tokenizer::Tokenizer;

pub type Token = u16;

pub trait TokenType: Copy + Eq + std::hash::Hash + Ord + TryFrom<usize> + 'static {
    fn to_index(self) -> usize;
    fn from_index(i: usize) -> Self;
    fn to_u32(self) -> u32;
    fn max_index() -> usize;
}

macro_rules! impl_token_type {
    ($($ty:ty),* $(,)?) => {
        $(
            impl TokenType for $ty {
                fn to_index(self) -> usize { self as usize }
                fn from_index(i: usize) -> Self { i as $ty }
                fn to_u32(self) -> u32 { self as u32 }
                fn max_index() -> usize { <$ty>::MAX as usize }
            }
        )*
    };
}

impl_token_type!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

#[cfg(test)]
mod tests {
    use std::{fs, mem::size_of, path::Path, time::Instant};

    use crate::{Token, prepare::*, tokenizer::Tokenizer};

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
            let regex_chunks: Vec<_> = re.find_iter(&ns).map(|mat| mat.as_str()).collect();
            assert_eq!(chunks, regex_chunks);
        }

        let ss = fs::read_to_string("shakespeare.txt").unwrap();
        let ns = normalize(&ss);
        let chunks = chunks(&ns);
        let regex_chunks: Vec<_> = re.find_iter(&ns).map(|mat| mat.as_str()).collect();
        assert_eq!(chunks, regex_chunks);
    }

    #[test]
    fn codepoint2token() {
        let tok: Tokenizer = Tokenizer::train(&STRINGS.concat(), &[], None);
        for &s in STRINGS.iter() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
        }
    }

    #[test]
    fn codepoint_bpe() {
        let tok: Tokenizer = Tokenizer::train(&STRINGS.concat(), &[], Some(512));
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
        let tok: Tokenizer = Tokenizer::train(&s, &[], Some(10000));
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
            enc.len() * size_of::<Token>()
        )
    }

    #[test]
    fn save_load() {
        let s = fs::read_to_string("big.txt").unwrap();
        let s0 = Instant::now();
        let start = Instant::now();
        let tok: Tokenizer = Tokenizer::train(&s, &["[PAD]", "[BOS]", "[EOS]"], Some(10000));
        println!("trained in {:?}", Instant::now() - start);
        let start = Instant::now();
        let enc = tok.encode(&s);
        println!("encoded in {:?}", Instant::now() - start);
        let start = Instant::now();
        assert!(tok.save(Path::new("tok.json")).is_ok());
        println!("saved in {:?}", Instant::now() - start);
        let start = Instant::now();
        let tok: Tokenizer = Tokenizer::load(Path::new("tok.json")).unwrap();
        println!("loaded in {:?}", Instant::now() - start);
        let start = Instant::now();
        let dec = tok.decode(&enc);
        println!("decoded in {:?}", Instant::now() - start);
        assert_eq!(s, dec);
        println!("total time {:?}", Instant::now() - s0);
        println!(
            "original byte len: {}, encoded byte len: {}",
            s.len(),
            enc.len() * size_of::<Token>()
        )
    }

    #[test]
    fn byte_fallback() {
        let tok: Tokenizer = Tokenizer::train(&STRINGS.concat(), &["[PAD]"], None);
        let s = "龍";
        // assert_eq!("<e9><be><8d>", tok.decode(&tok.encode(s)));
        assert_eq!(s, tok.decode(&tok.encode(s)));
    }
}
