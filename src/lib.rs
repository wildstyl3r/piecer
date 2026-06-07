use std::collections::{HashMap, HashSet};

pub type Token = u16;

pub struct Tokenizer {
    str2token: HashMap<String, Token>,
    token2str: HashMap<Token, String>,
    longest_str: usize,
}

impl Tokenizer {
    pub fn train(s: &str) -> Self {
        let mut alphabet = s
            .chars()
            .fold(HashSet::<String>::new(), |mut acc, c| {
                acc.insert(c.to_string());
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();
        alphabet.extend(["[UNK]", "[PAD]", "[BOS]", "[EOS]"].map(|s| s.to_string()));
        alphabet.sort();
        Self {
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
        }
    }

    pub fn encode(&self, mut s: &str) -> Vec<Token> {
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

    pub fn decode(&self, v: &[Token]) -> String {
        v.iter()
            .map(|t| {
                self.token2str
                    .get(t)
                    .unwrap_or(&self.token2str[&self.str2token["[UNK]"]])
                    .to_owned()
            })
            .collect::<Vec<_>>()
            .concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let strings = [
            "небо с утра дрожит\n\
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
            "---=== (((Hello, World!))) ===---\
            1000000 / 1000 = 1000",
            "Кружка-термос на 0.5л (вмещает 50/64 см³, вес 516г).\
            Скорость составила 90км/ч, а длина кабеля — 15мм. 
            В коробке 24шт. товара по цене 150руб/шт.\
            Это произошло в 90-х годах XX века. На 2-м этаже открылся новый офис. 
            В 10-12 часах езды от города находится заповедник. 
            Выпускники 11-го класса сдали экзамены на 95-100 баллов\
            Модель процессора: Intel Core i7-12700K или Эльбрус-8С.",
        ];
        let tok = Tokenizer::train(&strings.concat());
        for (i, &s) in strings.iter().enumerate() {
            let enc = tok.encode(s);
            let dec = tok.decode(&enc);
            assert_eq!(s, dec);
            println!(
                "{}: raw len is {} bytes, enc len is {} bytes {}",
                i,
                s.len(),
                enc.len() * 2,
                dec
            )
        }
    }
}
