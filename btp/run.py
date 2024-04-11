import copy
import collections
import random
import importlib
import logging
from time import time
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from trainer import zsre_trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from copy import deepcopy


LOG = logging.getLogger(__name__)

def run():

    # # T5-Small
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-ssm-nq")
    # tokenizer = AutoTokenizer.from_pretrained("google/t5-small-ssm-nq")
    # tokenize = tokenize_qa
    
    
    # # GPT-2
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2",padding_side='left')
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenize = tokenize_gpt
    
    # # Llama 2 directly from HuggingFace 
    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # # Llama 2 
    # tokenizer = LlamaTokenizer.from_pretrained("./models/llama-2-7b-hf")
    # model = LlamaForCausalLM.from_pretrained("./models/llama-2-7b-hf")
    
    # # Llama 2 Chat directly from HuggingFace 
    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenize = tokenize_gpt
    
    # # Llama 2 Chat
    # tokenizer = LlamaTokenizer.from_pretrained("./models/llama-2-7b-chat-hf")
    # model = LlamaForCausalLM.from_pretrained("./models/llama-2-7b-chat-hf")

    # # Mistral 7B
    # base_model = "mistralai/Mistral-7B-v0.1"
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,)
    # model = AutoModelForCausalLM.from_pretrained(
    #         base_model, torch_dtype=torch.bfloat16,quantization_config=bnb_config
    # )
    # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_eos_token = True
    # tokenizer.add_bos_token, tokenizer.add_eos_token
    # tokenize = tokenize_gpt
    
    # Phi-2
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenize = tokenize_phi
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    DEVICE1 = torch.device("cuda:0") #if torch.cuda.is_available() else torch.device("cpu")
    # DEVICE2 = torch.device("cuda:1") #if torch.cuda.is_available() else torch.device("cpu")
    copy_of_state_dict = deepcopy(model.state_dict()) 
    
    model.to(DEVICE1)
    
    # print(next(model.parameters()).is_cuda)
    
    # assert 1==2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    print('model size: {:.3f}GB'.format(size_all_mb))
    
    '''
    Load Dataset
    '''
    from dataset import NQ, zsRE, zsRE_balanced
    from metrics import F1_ACC, is_qa_error
    upstream = NQ()
    edits = zsRE_balanced(split="edit", n_edits=1000)
    # edit_holdouts = zsRE_balanced(split="holdout", n_edits=1000)

    '''Get Loaders
    '''
    batch_size = 1 #BTP
    edit_loader = DataLoader(edits, batch_size=batch_size, shuffle=False)
    # edit_holdout_loader = DataLoader(edit_holdouts, batch_size=batch_size, shuffle=False)
    upstream_loader = DataLoader(upstream, batch_size=100, shuffle=False)
    
    '''Define Metrics
    '''
    metric = F1_ACC # Measure QA F1
    is_error = is_qa_error
    

    from algs.zsre import qa

    alg = qa(model, tokenizer, copy_of_state_dict)
    # alg.to(DEVICE1)
    
    # assert 1==2

    trainer = zsre_trainer(alg,tokenize,metric,edit_loader,upstream_loader,edit_loader)
    
    file_path = "results_phi2.txt"
    for i in range(6):
        # continue
        torch.cuda.empty_cache()
        loc, es, g, mins = trainer.run_edit(num_steps = 2**i, opt = "Adam")
        with open(file_path, "a") as file:
            file.write("\n")
            file.write(f"num_steps = {2**i}, opt = Adam:" + "\n") 
            file.write(f"Locality: {loc}, Edit Success: {es}, Generality: {g}, Time: {mins} mins" + "\n")
    for i in range(6):
        # if i!=5:
        #     continue
        torch.cuda.empty_cache()
        loc, es, g, mins = trainer.run_edit(num_steps = 2**i, opt = "SGD")
        with open(file_path, "a") as file:
            file.write("\n")
            file.write(f"num_steps = {2**i}, opt = SGD:" + "\n") 
            file.write(f"Locality: {loc}, Edit Success: {es}, Generality: {g}, Time: {mins} mins" + "\n")

if __name__ == '__main__':
    run()