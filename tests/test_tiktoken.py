from app.tokenizer import tiktoken, version

def test_version():
    assert "0.12.0" == version("tiktoken")

def test_tiktoken_encode():
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text1 = "Hello do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|> ".join([text1, text2])

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    assert encoded == [15496, 466, 345, 588, 8887, 30, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 262, 20562, 13]


def test_tiktoken_decode():

    tokenizer = tiktoken.get_encoding("gpt2")


    decoded = tokenizer.decode([15496, 466, 345, 588, 8887, 30, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 262, 20562, 13])

    assert decoded == "Hello do you like tea?<|endoftext|> In the sunlit terraces of the palace."

def test_uknown_word():
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode("Akwirw ier", allowed_special={"<|endoftext|>"})

    assert encoded == [33901, 86, 343, 86, 220, 959]

