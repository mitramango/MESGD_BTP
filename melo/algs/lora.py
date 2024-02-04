from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os

from torch.nn import Parameter

from utils import *

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    MeloConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import LoraLayer, GraceLayer
from hooks import lora_backward_hook
# from models import BertClassifier
LOG = logging.getLogger(__name__)
def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)

class LORA(torch.nn.Module):
    def __init__(self, model, config, model_tok,scale=None):
        super(LORA, self).__init__()
        self.config = config

        '''Apply_lora
        '''

        ''' **BTP** Starts Comment

        r_num = config.grace.num_block * config.grace.num_rank_per_block
        self.lora_config = MeloConfig(
            r = r_num,
            lora_alpha = r_num,
            target_modules= list(config.model.target_modules),
            lora_dropout = config.lora.lora_dropout,
            task_type = config.lora_task_type,
            fan_in_fan_out= config.model.fan_in_fan_out,
            grace_layer = config.model.grace_layer,
            grace_config= OmegaConf.to_object(config.grace)
        )
        self.log_dict = {}

        '''#Load
        '''
        # self.original_model = model
        # self.model = model

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules()
        self.grace_layer = self.named_grace_layer()
        # self.register_lora_backward_hooks(lora_backward_hook)

        Ends Comment'''

        self.model = model

        '''Load Tokenizer
        '''
        self.model_tok = model_tok
        self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.lora.cls_name)

        '''Parameters to be optimized
        '''

        # ''' **BTP** optim parameters should be all model parameters now
        self.opt_params = self.optim_parameters()
        # self.opt_params = self.model.parameters()
        pass


    def optim_parameters(self):
        lora_params = self.model.parameters()
        return lora_params




    #TODO
    def load_from_checkpoint(self, save_path):
        print(save_path)


    def save_classifier_weights(self,cls_dir):
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir,exist_ok=True)
        torch.save(self.classifier.state_dict(),f"{cls_dir}/classifier.pt")
    def save_lora_weights(self,lora_dir):
        self.model.save_pretrained(lora_dir+"/lora_checkpoint")


    def reset_lora(self):
        for key in self.lora_list:
            self.model.get_submodule(key).reset_lora_parameters('default')

    def named_lora_modules(self):
        module_list = [key for key,_ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key),LoraLayer):
                lora_list.append(key)
        return lora_list

    def named_grace_layer(self) -> str:
        module_list = [key for key, _ in self.model.named_modules()]
        grace_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), GraceLayer):
                grace_list.append(key)
        assert len(grace_list) == 1, "At Most One Grace Layer"
        return grace_list[0]

    def register_lora_backward_hooks(self,backward_hook_fn):
        for key in self.lora_list:
            self.model.get_submodule(key).register_backward_hook(backward_hook_fn)


    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.disable_grace_layer()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()
        self.model.base_model.enable_grace_layer()

    ''' **BTP** changed the complete edit function
    def edit(self, tokens):
        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        setattr(self.model.get_submodule(self.grace_layer), "training", True)
        setattr(self.model.get_submodule(self.grace_layer), "edit_label", tokens["labels"])

        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            setattr(self.model.get_submodule(self.grace_layer), "batch_iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model.model(**tokens)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss # Log final loss

        setattr(self.model.get_submodule(self.grace_layer), "training", False)

    The NEW BTP EDIT function is given below'''
    def edit(self, tokens):
        optimizer = torch.optim.Adam(self.optim_parameters(), 1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        # setattr(self.model.get_submodule(self.grace_layer), "training", True)
        # setattr(self.model.get_submodule(self.grace_layer), "edit_label", tokens["labels"])

        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            # setattr(self.model.get_submodule(self.grace_layer), "batch_iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            # outputs = self.model.model(**tokens)
            outputs = self.model(**tokens)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss # Log final loss

        # setattr(self.model.get_submodule(self.grace_layer), "training", False)



    def generate(self, *args, **kwargs):
        # return self.model.model.generate(*args, **kwargs) BTP
        return self.model.generate(*args, **kwargs)

    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict










if __name__ == '__main__':
    pass


















