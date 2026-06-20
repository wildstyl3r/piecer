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
    pub fn pieces<'b>(
        &self,
        as_token: Token,
        tokens: &'b [String],
        protostack: &[ProtoToken],
    ) -> Vec<&'b str> {
        match self {
            ProtoToken::Pair(a, b) => {
                let mut result: Vec<&str> = Vec::new();
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
                        ProtoToken::Token(t) => result.push(&tokens[*t as usize]),
                    }
                }
                result
            }
            ProtoToken::Token(_) => vec![&tokens[as_token as usize]],
        }
    }
}
