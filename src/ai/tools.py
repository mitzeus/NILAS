def get_max_token_length_of_vocab(vocab, tokenizer):
    current_max = 0
    for word in vocab:
        tok = tokenizer.encode(word)
        if len(tok) > current_max:
            current_max = len(tok)

    return current_max
