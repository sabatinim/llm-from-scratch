from torch import tensor

from app.tokenizer import SimpleTokenizerV3, read_dataset
from app.data_sampling import create_dataloader

def test_sliding_window_approach():
    tokenizer = SimpleTokenizerV3()
    
    enc_sample = tokenizer.encode(text=read_dataset())

    context_size = 6

    print("")
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "--->", desired, "/",tokenizer.decode(context), "--->", tokenizer.decode([desired]))
      
def test_data_loader():
    data_loader = create_dataloader(read_dataset(), 
                                    batch_size=8,
                                    max_length=4,
                                    stride=4, 
                                    shuffle=False)
    iter_data_loader = iter(data_loader)
    batch_input, batch_target = next(iter_data_loader)
    print("")
    print(batch_input) 
    print(batch_target)
    assert len(data_loader) == 160
   