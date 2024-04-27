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
    

torch.save(model, "gpt2_nq_checkpoints/gpt_nq_checkpoint")

UP = [F1_ACC(gpt2, model, tokenize_gpt(e, tokenizer, DEVICE, test=True)) for e in
          iter(upstream_loader)]
UP_f1 = torch.tensor(UP).nanmean()

print (UP_f1.item())