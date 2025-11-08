import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from typing import Iterable, List
import os
from tqdm import tqdm

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

SRC_LANGUAGE_KEY = 'de'
TGT_LANGUAGE_KEY = 'en'

token_transform = {}
token_transform[SRC_LANGUAGE_KEY] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE_KEY] = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter: Iterable, language_key: str) -> List[str]:
    for data_sample in tqdm(data_iter, desc=f"Tokenizing for {language_key.upper()} vocab"):
        yield token_transform[language_key](data_sample['translation'][language_key])

def build_vocabs(train_iter):
    print("Building source language (DE) vocabulary...")
    src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, SRC_LANGUAGE_KEY),
                                          min_freq=1, specials=special_symbols, special_first=True)
    src_vocab.set_default_index(UNK_IDX)

    print("Building target language (EN) vocabulary...")
    tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter, TGT_LANGUAGE_KEY),
                                          min_freq=1, specials=special_symbols, special_first=True)
    tgt_vocab.set_default_index(UNK_IDX)
    
    print("Vocabularies built.")
    return src_vocab, tgt_vocab

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor([SOS_IDX] + src_sample + [EOS_IDX], dtype=torch.long))
        tgt_batch.append(torch.tensor([SOS_IDX] + tgt_sample + [EOS_IDX], dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

def text_transform(src_vocab, tgt_vocab):
    def transform(text_iter):
        for sample in text_iter:
            src_text = sample['translation'][SRC_LANGUAGE_KEY]
            tgt_text = sample['translation'][TGT_LANGUAGE_KEY]
            src_tokens = src_vocab(token_transform[SRC_LANGUAGE_KEY](src_text))
            tgt_tokens = tgt_vocab(token_transform[TGT_LANGUAGE_KEY](tgt_text))
            yield src_tokens, tgt_tokens
    return transform

def create_dataloaders(dataset_path: str, batch_size: int, subset_size: int = None, num_workers: int = 0):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    data_files = {
        'train': os.path.join(dataset_path, 'train.arrow'),
        'validation': os.path.join(dataset_path, 'validation.arrow'),
        'test': os.path.join(dataset_path, 'test.arrow')
    }
    
    for split, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} data file not found at {path}")

    hf_dataset = load_dataset('arrow', data_files=data_files)

    if subset_size is not None:
        train_iter = hf_dataset['train'].select(range(subset_size))
    else:
        train_iter = hf_dataset['train']
        
    src_vocab, tgt_vocab = build_vocabs(train_iter)
    
    print("Processing train data...")
    train_data = list(text_transform(src_vocab, tgt_vocab)(train_iter))
    print("Processing validation data...")
    valid_data = list(text_transform(src_vocab, tgt_vocab)(hf_dataset['validation']))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab
    