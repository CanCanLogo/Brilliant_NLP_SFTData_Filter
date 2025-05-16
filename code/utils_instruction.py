'''
    utils_QD_dataset: dataset formation & caching & loading programe.
    format dataset for QD model training & inferencing (may not)
    
'''
import re
import pandas as pd
import torch
import sys
import os
import json
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))

from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from utils_musique_transformers_dataset import load_data

dataset_dir = os.path.join(cur_dir, '../dataset')

'''
    format_raw_dataset: format raw dataset with the description, W/O evidence!!!
'''
def format_raw_dataset(raw_dataset):
    input_context = []
    output_question = []
    # def rep(matched):
    #     # format problem helper function
    #     ref_num = matched.group('ref').replace('#', '')
    #     return f'[[RQS]] {ref_num} [[RQE]]'
    for idx, item in enumerate(tqdm(raw_dataset)):
        # initial input_context, complex question only.
        in_context = item["instruction"] + item["input"]
        out_context = item["output"]
        # finally, generate the <END> for complete seq:
        input_context.append(in_context)
        output_question.append(out_context)
        pass
    return (input_context, output_question)
    pass

'''
    initialize_csv_dataset: cache csv dataset under dataset_dir, in train/valid.csv
'''
def initialize_csv_dataset(split):
    # raw_dataset = load_data(split)

    train_input_context, train_output_question = format_raw_dataset(raw_dataset['train'])
    valid_input_context, valid_output_question = format_raw_dataset(raw_dataset['validation'])
    # to DataFrame
    train_split = pd.DataFrame({'input_context': train_input_context, 'output_question': train_output_question})
    valid_split = pd.DataFrame({'input_context': valid_input_context, 'output_question': valid_output_question})
    # cache dataset (file):
    train_split.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False, sep='\t')
    valid_split.to_csv(os.path.join(dataset_dir, 'valid.csv'), index=False, sep='\t')
    pass


def dataset_tokenize_map(tokenizer, model_name):
    train_pth = 'train_1.json'
    train_pth = os.path.join(dataset_dir, train_pth)
    valid_pth = 'test_1.json'
    valid_pth = os.path.join(dataset_dir, valid_pth)
    dataset = load_dataset("json", data_files={'train':train_pth, 'validation':valid_pth})
    def _tokenize(batch_example):
        input_encodings = tokenizer(batch_example['input_context'], padding='max_length', max_length=48, truncation=True, return_tensors='pt')
        target_encodings = tokenizer(batch_example['output_question'], padding='max_length', max_length=160, truncation=True, return_tensors='pt')
        encodings = {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
            'decoder_attention_mask': target_encodings['attention_mask']
        }
        return encodings
        pass
    
    origin_columns = dataset['train'].features.keys()
    dataset['train'] = dataset['train'].map(_tokenize, batched=True, remove_columns=origin_columns)
    dataset['validation'] = dataset['validation'].map(_tokenize, batched=True, remove_columns=origin_columns)
    # set_format: set dataset columns type.
    # set_format is important! Or it will be list-like attribution.
    dataset['train'].set_format(type='torch')
    dataset['validation'].set_format(type='torch')
    # cache dataset: 
    torch.save(dataset, os.path.join(dataset_dir, f'{model_name}/dataset_infer.pt'))
    # cache tokenizer
    tokenizer.save_pretrained(os.path.join(dataset_dir, model_name))
    pass
    

'''
    load_tokenized_dataset: exposed interface for Quetion Decomposer model training.
'''
def load_tokenized_dataset(model_name, tokenizer):
    print(f'------loading tokenized dataset for {model_name}------')
    path = os.path.join(dataset_dir, f'{model_name}/dataset_infer.pt')
    for _ in range(2):
        try:
            dataset = torch.load(path)
        except Exception as e:
            # if model_name == 'bart-large':
            #     tokenizer = AutoTokenizer.from_pretrained(os.path.join(cur_dir, '../model/bart-large'))
                # try to add <END> to tokenizer
                # tokens = ['<END>']
            # tokenizer.add_special_tokens({'additional_special_tokens':['[[CQS]] 1 [[CQE]]', '[[CQS]] 2 [[CQE]]', '[[CQS]] 3 [[CQE]]', '[[CQS]] 4 [[CQE]]', '[[CQS]] 5 [[CQE]]', '[[RQS]] 1 [[RQE]]', '[[RQS]] 2 [[RQE]]', '[[RQS]] 3 [[RQE]]', '[[RQS]] 4 [[RQE]]', '[[RQS]] 5 [[RQE]]', '<end>']})
            # elif model_name == 't5-base':
            #     tokenizer = AutoTokenizer.from_pretrained(os.path.join(cur_dir, '../model/t5_decomposer'))
            #     tokenizer.add_special_tokens({'additional_special_tokens':["<END>"]})
            dataset_tokenize_map(tokenizer, model_name)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(dataset_dir, model_name))
    return tokenizer, dataset

if __name__ == '__main__':
    model_pth = os.path.join(cur_dir, '../model/flan-t5-large')
    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    model_name = 't5-large'
    load_tokenized_dataset(model_name, tokenizer) # debug