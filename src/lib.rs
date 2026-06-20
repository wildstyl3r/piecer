/*!
Unicode codepoint-based BPE tokenizer with byte fallback


*/
mod prepare;
pub mod tokenizer;

pub type Token = u16;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path, time::Instant};

    use crate::{prepare::*, tokenizer::Tokenizer};

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
    fn save_load() {
        let s = fs::read_to_string("big.txt").unwrap();
        let s0 = Instant::now();
        let start = Instant::now();
        let tok = Tokenizer::train(&s, Some(10000));
        println!("trained in {:?}", Instant::now() - start);
        let start = Instant::now();
        let enc = tok.encode(&s);
        println!("encoded in {:?}", Instant::now() - start);
        let start = Instant::now();
        assert!(tok.save(Path::new("tok.json")).is_ok());
        println!("saved in {:?}", Instant::now() - start);
        let start = Instant::now();
        let tok = Tokenizer::load(Path::new("tok.json")).unwrap();
        println!("loaded in {:?}", Instant::now() - start);
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
