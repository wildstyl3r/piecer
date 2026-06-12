piecer
======
An unicode codepoint based BPE tokenizer with byte fallback. The core vocabulary always includes full byte range (0x00-0xff) + any codepoints found in the training string.

At the encoding stage, any input that cannot be mapped to multi-byte vocabulary elements will be represented as a sequence of byte tokens. During the decoding stage, the tokenizer attempts to reconstruct valid, printable Unicode characters or prints hex code sequences like `<e9><e9><ff>` if it fails.

Relies on `priority-queue` and `aho-corasick` for pair merging and pattern matching.