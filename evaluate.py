import torch
import torch.nn as nn
import yaml
import argparse
import os
from tqdm import tqdm
import math
import warnings


from src.model import Transformer
from src.data import PAD_IDX, SOS_IDX, EOS_IDX

from torchmetrics.text import BLEUScore

warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_on_test_set(model, test_loader, criterion, device, tgt_vocab):
    model.eval()
    total_loss = 0
    bleu_score = BLEUScore(n_gram=4)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Evaluating on Test Set"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output_gold = tgt[:, :-1], tgt[:, 1:]

            src_mask = (src == PAD_IDX).unsqueeze(1).unsqueeze(2)
            tgt_padding_mask = (tgt_input == PAD_IDX).unsqueeze(1).unsqueeze(2)
            tgt_future_mask = torch.triu(torch.ones((tgt_input.shape[1], tgt_input.shape[1]), device=device), diagonal=1).bool()
            tgt_mask = tgt_padding_mask | tgt_future_mask.unsqueeze(0).unsqueeze(0)

            output = model(src, tgt_input, src_mask, tgt_mask)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output_gold.reshape(-1))
            total_loss += loss.item()

            pred_token_ids = output.argmax(2)
            tgt_itos = tgt_vocab.get_itos()
            
            for sentence_ids in pred_token_ids:
                tokens = [tgt_itos[i] for i in sentence_ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                all_preds.append(" ".join(tokens))
            
            for sentence_ids in tgt_output_gold:
                tokens = [tgt_itos[i] for i in sentence_ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                all_labels.append([" ".join(tokens)])

    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    bleu = bleu_score(all_preds, all_labels)

    return avg_loss, perplexity, bleu.item() * 100

def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained Transformer model on the test set.')
    parser.add_argument('--config', default='configs/optimized.yaml', help='Path to the config file.')
    parser.add_argument('--model_path', default='models/best_model.pt', help='Path to the model weights.')
    parser.add_argument('--src_vocab_path', default='models/src_vocab.pt', help='Path to the source vocabulary file.')
    parser.add_argument('--tgt_vocab_path', default='models/tgt_vocab.pt', help='Path to the target vocabulary file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

    from src.data import text_transform, collate_fn
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    print("\nLoading test data and vocabularies...")
    src_vocab = torch.load(args.src_vocab_path); tgt_vocab = torch.load(args.tgt_vocab_path)
    
    data_files = {'test': f"data/iwslt2017-en-de/test.arrow"}
    hf_dataset = load_dataset('arrow', data_files=data_files)
    test_data = list(text_transform(src_vocab, tgt_vocab)(hf_dataset['test']))
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn=collate_fn)
    print("Test data and vocabularies loaded successfully.")

    print("\nLoading model weights...")
    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'], n_layers=config['n_layers'],
        n_heads=config['n_heads'], d_ff=config['d_ff'], dropout=config['dropout']
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model weights loaded successfully.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    test_loss, test_ppl, test_bleu = evaluate_on_test_set(model, test_dataloader, criterion, device, tgt_vocab)

    print("\n" + "="*50)
    print(" Final Evaluation Report on Test Set")
    print("="*50)
    print(f"\tTest Loss      : {test_loss:.3f}")
    print(f"\tTest Perplexity: {test_ppl:.3f}")
    print(f"\tTest BLEU-4 Score: {test_bleu:.2f}")
    print("="*50)

if __name__ == '__main__':
    main()