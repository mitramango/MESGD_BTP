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

LOG = logging.getLogger(__name__)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    result = {}
    for d in list_of_dicts:
        for key, value in d.items():
            result.setdefault(key, []).append(value[0])
    return result

class zsre_trainer:
    def __init__(self, alg, tokenize, metric, edit_loader, upstream_loader, edit_holdout_loader):
        self.alg = alg
        self.tokenize = tokenize
        self.metric = metric
        self.edit_loader = edit_loader
        self.upstream_loader  = upstream_loader
        self.edit_holdout_loader = edit_holdout_loader
        self.batch_size = 1#config.grace.num_edit_per_block BTP

    def pre_editing_analyse(self):
        self.alg.disable_melo()

        with torch.no_grad():
            metric_dict = {'F1': [], 'ACC': []}
            for batch in iter(self.edit_loader):
                edit_input = self.tokenize(batch, self.alg.model_tok, DEVICE)
                f1, acc = self.metric(self.alg, edit_input)
                metric_dict['F1'].append(f1)
                metric_dict['ACC'].append(acc)
            original_f1 = torch.Tensor(metric_dict['F1']).nanmean()
            original_acc = torch.Tensor(metric_dict['ACC']).nanmean()
            LOG.info(
                f'Original average performance on edit set: F1: {original_f1.item():.4f} || ACC: {original_acc.item():.4f}')

            TRR_dict = {'F1': [], 'ACC': []}
            for up_batch in iter(self.upstream_loader):
                upstream_input = self.tokenize(up_batch, self.alg.model_tok, DEVICE)
                up_f1, up_acc = self.metric(self.alg, upstream_input)
                TRR_dict['F1'].append(up_f1)
                TRR_dict['ACC'].append(up_acc)
            upstream_f1 = torch.Tensor(TRR_dict['F1']).nanmean()
            upstream_acc = torch.Tensor(TRR_dict['ACC']).nanmean()
            LOG.info(
                f'Original average performance on upstream set: F1: {upstream_f1.item():.4f} || ACC: {upstream_acc.item():.4f}')

    def run_edit(self, num_steps = 16, opt = "Adam", model_name):
        # --- editing start ---
        ''' **BTP** commented self.alg.enable_melo()'''
        n_edits = 0
        batch_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_HOLDOUT = {}
        all_UP = {}
        all_VecDB = {}
        
        all_ES = 0
        all_hold = 0
        all_local = 0
        
        for i, batch in tqdm(enumerate(self.edit_loader)):
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            # print(batch[0])
            e = list_of_dicts_to_dict_of_lists(batch[1])
            tokens = self.tokenize(batch[0], self.alg.model_tok, DEVICE)
            n_edits += 1
            # --- perform edit ---
            edit_start = time()
            self.alg.edit(tokens, num_steps, opt)
            edit_time = time() - edit_start
            total_edit_time += edit_time

            # --- Compute and log metrics ---
            log_dict = {}
            with torch.no_grad():
                print(f'-------------------------    {n_edits}   ----------------------------------')
                # ES = [self.metric(self.alg, self.tokenize(batch[0], self.alg.model_tok, DEVICE))] #T5
                ES = [self.metric(self.alg, self.tokenize(batch[0], self.alg.model_tok, DEVICE, test=True), model_name)] #GPT2
                ES_f1 = torch.tensor([x[0] for x in ES]).nanmean()
                print(f'Batch {i} Edit Success: F1: {ES_f1}')
                all_ES = all_ES + ES_f1
                # assert 1==2
                
                holdout = [self.metric(self.alg, self.tokenize(e, self.alg.model_tok, DEVICE, test=True), model_name)]
                holdout_f1 = torch.tensor([x[0] for x in holdout]).nanmean()
                all_hold = all_hold + holdout_f1
                print(f'Batch {i} Generality after Editing: F1: {holdout_f1}')
                holdout_acc = torch.tensor([x[1] for x in holdout]).nanmean()

                
                # UP = [self.metric(self.alg, self.tokenize(e, self.alg.model_tok, DEVICE, test=True)) for e in
                      # iter(self.upstream_loader)]
                UP_f1 = torch.tensor([0.00]).nanmean()
                all_local = all_local + UP_f1
                print(f'Batch {i} Locality after Editing: F1: {UP_f1}')
                UP_acc = torch.tensor([0.00]).nanmean()
                
                # --- Log metrics and push to Weights & Biases ---
                log_dict["loc"] = {'locality_f1': UP_f1.item()}  # Locality
                log_dict["ES"] = {'ES_f1': ES_f1.item()}  # Edit Success
                log_dict["train_time"] = edit_time / 60  # Time it takes to make one edit
                log_dict["n_edits"] = n_edits  # Raw edit label
                log_dict['generality'] = {'generality_f1': holdout_f1.item()}
                
                print(f"Number of edits {n_edits}")
                
                for k in log_dict:
                    LOG.info(f"[+eval result+]{k}: {log_dict[k]}")

                all_UP[n_edits] = log_dict["loc"]
                all_HOLDOUT[n_edits] = log_dict["generality"]
                all_edit_time[n_edits] = total_edit_time

            self.alg.model.load_state_dict(self.alg.orig_state_dict)
        print('Average Locality: ', (all_local/n_edits).item(), 'Average Edit Success: ', (all_ES/n_edits).item() , 'Average Generality: ', (all_hold/n_edits).item())
        
        return (all_local/n_edits).item(), (all_ES/n_edits).item(), (all_hold/n_edits).item(), total_edit_time / 60

        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")


