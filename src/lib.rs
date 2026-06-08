use std::collections::{HashMap, HashSet};

use fancy_regex::{Captures, Regex};

pub type Token = u16;

pub struct Tokenizer {
    str2token: HashMap<String, Token>,
    token2str: HashMap<Token, String>,
    longest_str: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum GroupType {
    Letters,
    Digits,
    Spaces,
    Other,
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
            token2str: alphabet
                .iter()
                .enumerate()
                .map(|(i, t)| (i as Token, t.to_owned()))
                .collect(),
            str2token: alphabet
                .iter()
                .enumerate()
                .map(|(i, t)| (t.to_owned(), i as Token))
                .collect(),
            longest_str: alphabet.iter().fold(0, |l, s| std::cmp::max(l, s.len())),
        };

        if let Some(vocab_size) = vocab_size {
            let mut chunks: Vec<_> = Tokenizer::chunks(&Tokenizer::normalize(s))
                .iter()
                .map(|str_chunk| tokenizer.encode_normalized(str_chunk))
                .collect();
            while tokenizer.token2str.len() < vocab_size {
                let mut counts = HashMap::new();
                let mut top_key = None;
                for (i, chunk) in chunks.iter().enumerate() {
                    for (j, current_pair) in chunk.windows(2).enumerate() {
                        let e = counts
                            .entry((current_pair[0], current_pair[1]))
                            .or_insert(HashSet::new());
                        e.insert((i, j));
                        let current_count = e.len();
                        top_key = match top_key {
                            Some((a, b)) => {
                                if counts[&(a, b)].len() < current_count {
                                    Some((current_pair[0], current_pair[1]))
                                } else {
                                    Some((a, b))
                                }
                            }
                            None => Some((current_pair[0], current_pair[1])),
                        };
                    }
                }
                match top_key {
                    Some((a, b)) => {
                        let merge_str = tokenizer.token2str.get(&a).unwrap().to_string()
                            + tokenizer.token2str.get(&b).unwrap();
                        let token = tokenizer.token2str.len() as Token;
                        tokenizer.longest_str =
                            std::cmp::max(tokenizer.longest_str, merge_str.len());
                        tokenizer.token2str.insert(token, merge_str.clone());
                        tokenizer.str2token.insert(merge_str, token);

                        let mut positions: Vec<_> = counts.get(&(a, b)).unwrap().iter().collect();
                        positions.sort_by_key(|(_chunk, _position)| _position);
                        for (chunk, position) in positions.into_iter().rev() {
                            chunks[*chunk].splice(position..&(position + 2), [token]);
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
            .replace_all(s, "$1$2 ")
            .to_string();
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
        Tokenizer::denormalize(
            &v.iter()
                .map(|t| {
                    self.token2str
                        .get(t)
                        .unwrap_or(&self.token2str[&self.str2token["[UNK]"]])
                        .to_owned()
                })
                .collect::<Vec<_>>()
                .concat(),
        )
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn codepoint2token() {
        let strings = [
            //П. Никулин (без названия?), W. Blake (Auguries of Innocence), В. Пелевин (Омон Ра)
            "небо с утра дрожит\n\
            цвета распадаются на базовые и на те что недоступны глазу\n\
            в парке снова пахнет слезоточивым газом\n\
            кислотой\n\
            порохом\n\
            известью\n\
            выбешивает дветысячидевятнадцатый и скоро совсем выбесит<...>\
            \n\nTo see a world in a grain of sand\n\
            And a heaven in a wild flower,\n\
            Hold infinity in the palm of your hand\n\
            And eternity in an hour",
            "\n\nМы летели со скоростью двух с половиной километров в секунду, и инерционная часть полета заняла около трех суток, \
            но у меня осталось чувство, что я летел не меньше недели. Наверно, потому, что солнце несколько раз в сутки проходило \
            перед глазками, и каждый раз я любовался восходом и закатом небывалой красоты.\nОт огромной ракеты теперь оставался \
            только лунный модуль, состоявший из ступени коррекции и торможения, где сидел Дима Матюшевич, и спускаемого аппарата, \
            то есть попросту лунохода на платформе. Чтоб не тратить лишнее горючее, обтекатель отстрелился еще перед разгоном с \
            орбиты спутника, и за бортом лунохода теперь был открытый космос.",
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
        let tok = Tokenizer::train(&strings.concat(), None);
        for &s in strings.iter() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
        }
    }

    #[test]
    fn codepoint_bpe() {
        let strings = [
            "пробелы    пробелы[UNK]небо с утра дрожит\n\
            цвета распадаются на базовые и на те что недоступны глазу\n\
            в парке снова пахнет слезоточивым газом\n\
            кислотой\n\
            порохом\n\
            известью\n\
            выбешивает дветысячидевятнадцатый и скоро совсем выбесит...\
            \n\nTo see a world in a grain of sand\n\
            And a heaven in a wild flower,\n\
            Hold infinity in the palm of your hand\n\
            And eternity in an hour.",
            "\n\nМы летели со скоростью двух с половиной километров в секунду, и инерционная часть полета заняла около трех суток, \
            но у меня осталось чувство, что я летел не меньше недели. Наверно, потому, что солнце несколько раз в сутки проходило \
            перед глазками, и каждый раз я любовался восходом и закатом небывалой красоты.\nОт огромной ракеты теперь оставался \
            только лунный модуль, состоявший из ступени коррекции и торможения, где сидел Дима Матюшевич, и спускаемого аппарата, \
            то есть попросту лунохода на платформе. Чтоб не тратить лишнее горючее, обтекатель отстрелился еще перед разгоном с \
            орбиты спутника, и за бортом лунохода теперь был открытый космос.",
            "кто-то где-то что-то с чем-то смешивает- смешивает да ка-а-ак смешает",
            "---=== (((Hello, World!))) ===---\n\
            1000000 / 1000 = 1000",
            "Кружка-термос на 0.5л (вмещает 50/64 см³, вес 516г).\
            Скорость составила 90км/ч, а длина кабеля — 15мм. 
            В коробке 24шт. товара по цене 150руб/шт.\
            Это произошло в 90-х годах XX века. На 2-м этаже открылся новый офис. 
            В 10-12 часах езды от города находится заповедник. 
            Выпускники 11-го класса сдали экзамены на 95-100 баллов\
            Модель процессора: Intel Core i7-12700K или Эльбрус-8С.",
        ];
        let tok = Tokenizer::train(&strings.concat(), Some(512));
        for &s in strings.iter() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
        }
    }
}
