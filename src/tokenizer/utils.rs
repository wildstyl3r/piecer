use serde::{Deserialize, Serialize};

use crate::Token;

#[derive(Serialize)]
pub(crate) struct Export<'a> {
    pub version: &'static str,
    pub vocabulary: &'a [String],
}

#[derive(Deserialize)]
pub(crate) struct Import {
    pub version: String,
    pub vocabulary: Vec<String>,
}

pub(crate) enum ProtoToken {
    Pair(usize, usize),
    Token(Token),
}

impl ProtoToken {
    pub fn pieces(&self, as_token: Token, tokens: &[String], protostack: &[ProtoToken]) -> String {
        match self {
            ProtoToken::Pair(a, b) => {
                let mut result = String::new();
                let mut stack = Vec::new();
                stack.push(b);
                stack.push(a);
                while !stack.is_empty() {
                    let current = &protostack[*stack.pop().unwrap()];
                    match current {
                        ProtoToken::Pair(a, b) => {
                            stack.push(b);
                            stack.push(a);
                        }
                        ProtoToken::Token(t) => result += &tokens[*t as usize],
                    }
                }
                result
            }
            ProtoToken::Token(_) => String::from(&tokens[as_token as usize]),
        }
    }
}
