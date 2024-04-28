import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import NQ
from utils import tokenize_gpt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train GPT2 for Question Answering on NQ dataset')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--l', type=int, default=106926, help='Value for l')
args = parser.parse_args()

learning_rate = args.learning_rate
epochs = args.epochs
l_value = args.l

upstream = NQ(l=l_value)
upstream_loader = DataLoader(upstream, batch_size=1, shuffle=False)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
    
for e in tqdm(range(epochs)):
    
    model.train()
    for batch in upstream_loader:
        tokens = tokenize_gpt(batch, tokenizer, DEVICE)

        outputs = model(**tokens)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model.eval()
    val_loss = 0
    cnt = 0
    for batch in upstream_loader:
        tokens = tokenize_gpt(batch, tokenizer, DEVICE)

        outputs = model(**tokens)
        loss = outputs.loss
        val_loss += loss.detach().cpu().numpy()
        cnt += 1
    
    print(f"Epoch {e}, Loss: {val_loss / cnt}")
    

torch.save(model.state_dict(), "gpt2_nq_checkpoints/gpt_nq_checkpoint.pth")


def F1_ACC(model_name, alg, batch, tok):
    try:
        if "t5" in model_name:
            # T5
            preds = alg.generate(batch["input_ids"], max_length=20).squeeze(1) 
        else:
            # # GPT2
            preds = alg.generate(batch["input_ids"], pad_token_id=tok.pad_token_id, max_new_tokens=20).squeeze(1) 
            preds = [preds[i][len(batch["input_ids"][i]):] for i in range(len(preds))]
        
        f1 = F1(preds, batch, tok)
        acc = 1.0
        return f1, acc
    except Exception as e:
        raise e

def F1(preds, batch, tok):
    # try:
    # print("this", len(batch["labels"]))
    try:
        f1_list = []
        # print("yahan", len(preds)) 
        for p, g in zip(preds,batch["labels"]):
            p = p[p !=  tok.pad_token_id].cpu().squeeze()
            g = g[g != -100].cpu().squeeze()  # -100 might be nonsense
            #i print(p, g)
            num_same = len(np.intersect1d(p, g))
            len_pred = len(p)
            len_gold = len(g)
            precision = num_same / len_pred
            recall = 1.0 * num_same / len_gold
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
    except:
        # print("galti")
        return 0.
    # print(f1_list)
    return sum(f1_list) / len(f1_list)

UP = [F1_ACC("gpt2", model, tokenize_gpt(e, tokenizer, DEVICE, test=True), tokenizer) for e in
          iter(upstream_loader)]
UP_f1 = torch.tensor(UP).nanmean()

print (UP_f1.item())