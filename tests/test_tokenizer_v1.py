from app.tokenizer import SimpleTokenizerV1, tokenize, read_dataset

def test_simple_tokenizer_encode():
    tokenizer = SimpleTokenizerV1(vocab=tokenize(read_dataset()))

    encoded = tokenizer.encode(
        """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")
    assert encoded == [4409, 4680, 4681, 4647, 3870, 4606, 4508, 4673, 4321, 4223, 4673, 4689, 4478, 4688, 2923, 4383, 4641, 1968, 2180, 4688]

def test_simple_tokenizer_decode():
    tokenizer = SimpleTokenizerV1(vocab=tokenize(read_dataset()))
    ids = [4409, 4680, 4681, 4647, 3870, 4606, 4508, 4673, 4321, 4223, 4673, 4689, 4478, 4688, 2923, 4383, 4641, 1968, 2180, 4688]
    decoded = tokenizer.decode(ids)
    
    assert decoded == """It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

