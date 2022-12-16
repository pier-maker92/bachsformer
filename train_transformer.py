import numpy as np 
from tqdm import tqdm
from vq_vae.tools import * 
from transformer_decoder_only.model import GPT
from transformer_decoder_only.trainer import Trainer

class CodebooksDataset(Dataset):
    def __init__(self,device):
        codebooks = self.get_codebooks()
        self.data = np.array(codebooks)
        self.device = device
        self.length = len(codebooks[0])
    
    def get_codebooks(self):
        f = open("data/vocab/vocab_16_192length.txt", "r")
        codebooks_idx = []
        for line in f.readlines():
            line = [int(i) for i in (line.split("\n")[0].split(',')) if i]
            #line.insert(0,-1)
            codebooks_idx.append(line)
        return codebooks_idx
    
    def __len__(self):
        return len(self.data)
    
    def get_vocab_size(self):
        return 16
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length-1

    def __getitem__(self, idx):
        # the inputs to the transformer will be the offset sequence
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y
    
    def shuffle_it(self):
        np.random.shuffle(self.data)

def batch_end_callback(trainer):
    print(f"\riter {trainer.iter_num}: train loss {trainer.loss.item():.5f}",end="")
    torch.save(model.state_dict(),"bachsformer")

if __name__=="__main__":
    dtype = torch.float
    device = torch.device("mps")
    config_name = "ludovico-mini"
    ludovico_vae = LudovicoVAE(config_name)
    # get model
    try:
        model = ludovico_vae.get_model()
    except:
        print(f"No model found with this configuration: {config_name}")
    # get vocab
    ludovico_vae.codebooks2vocab(model)
    del model # get rid of VQ-VAE, no longer needed
    train_dataset = CodebooksDataset(device)
    
    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-bach'
    print(f"vocab_size: {train_dataset.get_vocab_size()}")
    print(f"block_size: {train_dataset.get_block_size()}")
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    #model
    model = GPT(model_config).to(device)
    model_name = "bachsformer"

    # create a Trainer object
    batch_size = 128
    steps_per_epoch = train_dataset.__len__()//batch_size
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = steps_per_epoch*500
    train_config.num_workers = 0
    train_config.device=device
    train_config.batch_size = batch_size
    trainer = Trainer(train_config, model, train_dataset)
    try:
        model.load_state_dict(torch.load(model_name))
        print("model loaded from pretrained")
    except: pass

    # train
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()