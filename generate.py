import random
import numpy as np 
from tqdm import tqdm
from vq_vae.tools import * 
from transformer_decoder_only.model import GPT
from transformer_decoder_only.utils import set_seed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('-t','--temperature', type=float, default=.9)
parser.add_argument('-tk','--top_key', type=int, default=16)
parser.add_argument('-s','--seed', type=int, default=42)

def generate_from_idx(model,idx,seed=None):
    print(idx)
    model.eval()
    num_embeddings = model._vq_vae._num_embeddings
    embedding = model._vq_vae._embedding

    if seed: torch.random.manual_seed(seed)
    encoding_indices = torch.tensor(idx,device=device).unsqueeze(1)#torch.randint(0, 17, (16,1), device=device)
    encodings = torch.zeros(encoding_indices.shape[0], num_embeddings, device=device)
    encodings.scatter_(1, encoding_indices, 1)
    
    # Quantize and unflatten
    quantized = torch.matmul(encodings, embedding.weight).view(torch.Size([1, 8, 2, 128]))
    quantized = quantized.detach().permute(0, 3, 1, 2).contiguous()
    generated = model._decoder(quantized)
    return generated

if __name__=="__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #set_seed(args.seed) # not working
    dtype = torch.float
    device = torch.device("cpu")
    config_name = "ludovico-mini"
    ludovico_vae = LudovicoVAE(config_name,device=device)
    # get model
    try:
        model = ludovico_vae.get_model()
    except:
        print(f"No model found with this configuration: {config_name}")

    # create a GPT instance
    gpt_model_config = GPT.get_default_config()
    gpt_model_config.model_type = 'gpt-bach'
    #TODO recal from model config
    gpt_model_config.vocab_size = 16  # 16 codebooks 
    gpt_model_config.block_size = 191 # 16 codebooks 
    # load model
    gpt_model = GPT(gpt_model_config).to(device)
    gpt_model_name = "bachsformer"
    try:
        gpt_model.load_state_dict(torch.load(gpt_model_name))
        print("model loaded from pretrained")
    except: pass
    # generate codebooks index with transformers
    gpt_model.eval()
    if not args.file:
        first_idx = random.randint(0,15)
        x = torch.tensor([first_idx],device=device).unsqueeze(0)
        generated = []
        for k in range(16):
            if not k:
                codebooks_idx = gpt_model.generate(x, 191, do_sample=True, top_k=args.top_key, temperature=args.temperature)
            else: codebooks_idx = gpt_model.generate(x, 176, do_sample=True, top_k=args.top_key, temperature=args.temperature)
            codebooks_idx = codebooks_idx.data.cpu().detach().numpy().squeeze()
            x = torch.tensor(codebooks_idx[-16:],device=device).unsqueeze(0)
            print(x)
            quarters  = np.array([codebooks_idx[i*16:i*16+16] for i in range(8)])
            for q in quarters:
                generated.append(q)
    else:
        file_name = args.file.split(".")[0]
        first_idx = f"from_input_{file_name}"
        x = torch.tensor(ludovico_vae.codebooks2vocab(model,tune_name = args.file),device=device).unsqueeze(0)
        generated = []
        for k in range(16):
            if not k:
                codebooks_idx = gpt_model.generate(x, 192-x.shape[0], do_sample=True, top_k=args.top_key, temperature=args.temperature)
            else: codebooks_idx = gpt_model.generate(x, 176, do_sample=True, top_k=args.top_key, temperature=args.temperature)
            codebooks_idx = codebooks_idx.data.cpu().detach().numpy().squeeze()
            x = torch.tensor(codebooks_idx[-16:],device=device).unsqueeze(0)
            print(x)
            quarters  = np.array([codebooks_idx[i*16:i*16+16] for i in range(8)])
            for q in quarters:
                generated.append(q)

    bars_generated = []
    for c in generated:
        new = np.round(generate_from_idx(model,c,seed=None).data.cpu().detach().numpy().squeeze())
        bars_generated.append(new)

    miniaturizer = MidiMiniature(1) # 1/4th
    gen = miniaturizer.miniature2midi(bars_generated)
    temp = "_".join(str(args.temperature).split("."))
    name = f"data/generated/gen_{first_idx}_temperature_{temp}_top_key_{args.top_key}.mid"
    gen.save(name)
    print(f"\nstored with name : {name}\n")
            

    