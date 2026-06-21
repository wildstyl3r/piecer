use serde::{Deserialize, Serialize};

use crate::TokenType;

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

pub(crate) enum ProtoToken<T> {
    Pair(usize, usize),
    Token(T),
}

impl<T: TokenType> ProtoToken<T> {
    pub fn pieces(&self, as_token: T, tokens: &[String], protostack: &[ProtoToken<T>]) -> String {
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
                        ProtoToken::Token(t) => result += &tokens[t.to_index()],
                    }
                }
                result
            }
            ProtoToken::Token(_) => String::from(&tokens[as_token.to_index()]),
        }
    }
}
