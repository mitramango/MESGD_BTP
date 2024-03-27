import copy
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

LOG = logging.getLogger(__name__)

def run():

    # # T5-Small
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-ssm-nq")
    # tokenizer = AutoTokenizer.from_pretrained("google/t5-small-ssm-nq")
    # tokenize = tokenize_qa
    
    
    # GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenize = tokenize_gpt
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(DEVICE)
    

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
    
    alg = qa(model, tokenizer)
    alg.to(DEVICE)

    trainer = zsre_trainer(alg,tokenize,metric,edit_loader,upstream_loader,edit_loader)
    
    file_path = "results_gpt2.txt"
    for i in range(6):
        torch.cuda.empty_cache()
        loc, es, g, mins = trainer.run_edit(num_steps = 2**i, opt = "Adam")
        with open(file_path, "a") as file:
            file.write("\n")
            file.write(f"num_steps = {2**i}, opt = Adam:" + "\n") 
            file.write(f"Locality: {loc}, Edit Success: {es}, Generality: {g}, Time: {mins} mins" + "\n")
    for i in range(6):
        torch.cuda.empty_cache()
        loc, es, g, mins = trainer.run_edit(num_steps = 2**i, opt = "SGD")
        with open(file_path, "a") as file:
            file.write("\n")
            file.write(f"num_steps = {2**i}, opt = SGD:" + "\n") 
            file.write(f"Locality: {loc}, Edit Success: {es}, Generality: {g}, Time: {mins} mins" + "\n")

if __name__ == '__main__':
    run()