piecer
======
An unicode codepoint based BPE tokenizer with byte fallback. The core vocabulary always includes full byte range (0x00-0xff) and special tokens provided, then optionally other codepoints found in the training string plus BPE merges.

At the encoding stage, any input that cannot be mapped to multi-byte vocabulary elements will be represented as a sequence of byte tokens. During the decoding stage, the tokenizer attempts to reconstruct valid, printable Unicode characters or prints hex code sequences like `<e9><e9><ff>` if it fails.

Relies on `priority-queue` and `daachorse` for pair merging and pattern matching, uses `rustc-hash`'s FxHashMap for pair-wise counters and position querying.