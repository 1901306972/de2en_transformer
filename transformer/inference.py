import torch
import yaml
import spacy
import argparse
import os
from src.model import Transformer 
from src.data import (
    UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX,
    SRC_LANGUAGE_KEY, TGT_LANGUAGE_KEY
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

def translate_sentence(model, sentence: str, src_vocab, tgt_vocab, token_transform, device, max_len=50):
    model.eval()

    tokens = [tok.text for tok in token_transform[SRC_LANGUAGE_KEY](sentence.lower())]
    
    text_to_indices = [SOS_IDX] + [src_vocab.get_stoi().get(token, UNK_IDX) for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    src_mask = (src_tensor == PAD_IDX).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        memory = model.encoder(src_tensor, src_mask)

    tgt_tokens_indices = [SOS_IDX]
    for i in range(max_len - 1):
        tgt_tensor = torch.LongTensor(tgt_tokens_indices).unsqueeze(0).to(device)
        
        tgt_padding_mask = (tgt_tensor == PAD_IDX).unsqueeze(1).unsqueeze(2)
        tgt_future_mask = torch.triu(torch.ones((tgt_tensor.shape[1], tgt_tensor.shape[1]), device=device), diagonal=1).bool()
        tgt_mask = tgt_padding_mask | tgt_future_mask.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            decoder_output = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
            logits = model.generator(decoder_output[:, -1])
            
            pred_token_idx = logits.argmax(1).item()
        
        tgt_tokens_indices.append(pred_token_idx)

        if pred_token_idx == EOS_IDX:
            break
            
    tgt_itos = tgt_vocab.get_itos()
    translated_tokens = [tgt_itos[i] for i in tgt_tokens_indices]
    
    return " ".join(translated_tokens[1:-1])

def main():
    parser = argparse.ArgumentParser(description='Inference with a trained Transformer model.')
    parser.add_argument('--config', default='configs/ultimate.yaml', help='Path to config file.')
    parser.add_argument('--model_path', default='models/best_model.pt', help='Path to the trained model weights.')
    parser.add_argument('--src_vocab_path', default='models/src_vocab.pt', help='Path to the source vocabulary file.')
    parser.add_argument('--tgt_vocab_path', default='models/tgt_vocab.pt', help='Path to the target vocabulary file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading vocabularies from files...")
    src_vocab = torch.load(args.src_vocab_path)
    tgt_vocab = torch.load(args.tgt_vocab_path)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    print("Vocabularies loaded successfully.")

    print("\nLoading Spacy tokenizers...")
    token_transform = {
        SRC_LANGUAGE_KEY: spacy.load('de_core_news_sm'),
        TGT_LANGUAGE_KEY: spacy.load('en_core_web_sm')
    }
    print("Tokenizers loaded successfully.")
    
    print("\nLoading model weights...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'], n_layers=config['n_layers'],
        n_heads=config['n_heads'], d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model weights loaded successfully.")

    print("\n" + "="*50)
    print("Interactive translation ready. Type a German sentence or 'quit' to exit.")
    print("="*50)
    while True:
        try:
            sentence = input("German > ")
            if sentence.lower() == 'quit':
                break
            if not sentence:
                continue
            translation = translate_sentence(model, sentence, src_vocab, tgt_vocab, token_transform, device)
            print(f"English < {translation}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == '__main__':
    main()
    