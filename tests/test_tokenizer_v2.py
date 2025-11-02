from app.tokenizer import SimpleTokenizerV2, read_dataset, tokenize

def test_simple_tokenizer_encode_unknown_word():
    tokenizer = SimpleTokenizerV2(vocab=tokenize(read_dataset()))

    text1 = "Hello do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|> ".join([text1, text2])

    encoded = tokenizer.encode(text)

    assert encoded == [4691, 4040, 4321, 4562, 669, 4629, 4690, 1572, 4647, 1154, 650, 4686, 4647, 4691, 4688]

def test_simple_tokenizer_decode_unknown_word():
    tokenizer = SimpleTokenizerV2(vocab=tokenize(read_dataset()))

    decoded = tokenizer.decode([4691, 4040, 4321, 4562, 669, 4629, 4690, 1572, 4647, 1154, 650, 4686, 4647, 4691, 4688])


    assert decoded == "<|unk|> do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."

