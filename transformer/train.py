import torch
import torch.nn as nn
import yaml
import time
import random
import numpy as np
import os
import math
import argparse
import warnings
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.model import Transformer
from src.data import create_dataloaders, PAD_IDX


warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

def set_seed(seed):
    
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def create_mask(src, tgt, pad_idx, device):
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_padding_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_future_mask = torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=device), diagonal=1).bool()
    final_tgt_mask = tgt_padding_mask | tgt_future_mask.unsqueeze(0).unsqueeze(0)
    return src_mask, final_tgt_mask

def run_epoch(is_training: bool, model, loader, criterion, device, optimizer=None):
    model.train(is_training)
    epoch_loss = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    desc = "Training" if is_training else "Validating"
    
    with context:
        for src, tgt in tqdm(loader, desc=desc):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            src_mask, tgt_mask = create_mask(src, tgt_input, PAD_IDX, device)
            
            if is_training:
                assert optimizer is not None
                optimizer.zero_grad()
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='Train a Transformer model.'); 
    parser.add_argument('--config', default='configs/optimized.yaml', help='Path to config file.'); 
    args = parser.parse_args()
    
    print(f"Loading config: {args.config}");
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    set_seed(config['seed']); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    
    print("\nStarting training with FIXED learning rate on a SUBSET of data.")
    train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
        dataset_path="data/iwslt2017-en-de", 
        batch_size=config['batch_size'], 
        # subset_size=50000
    )
    print(f"Vocab sizes -> Src: {len(src_vocab)}, Tgt: {len(tgt_vocab)}")

    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), 
        d_model=config['d_model'], n_layers=config['n_layers'], n_heads=config['n_heads'], 
        d_ff=config['d_ff'], dropout=config['dropout']
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
        print(f"Using {torch.cuda.device_count()} GPUs for Data Parallel!"); 
        model = nn.DataParallel(model)
    model.to(device)

    module_to_init = model.module if isinstance(model, nn.DataParallel) else model
    for p in module_to_init.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    print(f"Using AdamW optimizer with fixed learning rate: {config['learning_rate']}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=config.get('label_smoothing', 0.0))
    print(f"Using Label Smoothing: {config.get('label_smoothing', 0.0)}")

    best_val_loss = float('inf'); train_losses, val_losses = [], []
    patience = config.get('patience', 5); epochs_no_improve = 0; print(f"Early stopping enabled with patience: {patience}")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        train_loss = run_epoch(True, model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(False, model, val_loader, criterion, device)
        
        end_time = time.time()
        
        train_losses.append(train_loss); val_losses.append(val_loss)
        mins, secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch+1:02} | Time: {mins:.0f}m {secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f}, PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal. Loss: {val_loss:.3f}, PPL: {math.exp(val_loss):7.3f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, 'models/best_model.pt'); 
            print(f"----> New best model saved (Val Loss: {best_val_loss:.3f})")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break
    
    print("\n" + "="*50); print("Saving final assets...")
    torch.save(src_vocab, 'models/src_vocab.pt'); torch.save(tgt_vocab, 'models/tgt_vocab.pt')
    
    plt.figure(figsize=(10, 6)); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Validation Loss');
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training and Validation Loss Curves'); plt.grid(True)
    plt.savefig('results/loss_curves_final3.png');
    print("All assets saved. Training complete.")

if __name__ == '__main__':
    main()