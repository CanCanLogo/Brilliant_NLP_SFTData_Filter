from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, T5Tokenizer, T5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR 
import torch.optim as optim
# from torch_optimizer import Adafactor
# import deepspeed

import os
from tqdm import tqdm
# from musique_transformers_dataset import get_dataset
import sys
sys.path.append('./utils')
from utils.utils_instruction import load_tokenized_dataset
import json

# AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'

os.environ["LOCAL_RANK"] = "0"

os.environ["WORLD_SIZE"] = "1"

# deepspeed.ops.op_builder.CPUAdamBuilder().load()

cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).cuda()
    return tokenizer, model

def resize_model_embedding(origin_len, new_tokenizer, model):
    # adjust embeddings.
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:origin_vocab_len].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:origin_vocab_len].mean(dim=0, keepdim=True)
    if model_name == 't5-large':
        input_embeddings[origin_vocab_len:] = input_embeddings_avg
        output_embeddings[origin_vocab_len:] = output_embeddings_avg
    elif model_name == 't5-base':
        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg

    # model.set_input_embeddings(input_embeddings)
    # model.set_output_embeddings(output_embeddings)
    pass
def decomposition_generation(tokenizer, model, iter, output_path):
    result = []
    with torch.no_grad():
        for batch in tqdm(iter):
            # print(batch)
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            # reference = batch['reference']
            # question = batch['question']
            # try to use beam search as baseline
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=160, num_beams=10)
            ques = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            cur_res = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            result.extend(list(zip(ques, cur_res)))
            # break
    with open(output_path, 'w') as f:  
        json.dump(result, f, indent=4)
        # torch.save({'decomposition': result}, output_path)

if __name__ == '__main__':
    # init distributed envs:
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # deepspeed.init_distributed(dist_backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    model_name = 't5-large'
    model_path = 'model/t5_decomposer' if model_name=='t5-base' else 'model/flan-t5-large'
    # tokenizer, model = load_model(model_path)

    tokenizer = T5Tokenizer.from_pretrained(os.path.join(cur_dir, model_path))
    # model = T5ForConditionalGeneration.from_pretrained(os.path.join(cur_dir, model_path)).cuda()
    tokenizer, dataset = load_tokenized_dataset(model_name, tokenizer)

    model_path = os.path.join(cur_dir, 'model/flan-t5-large')
    tokenizer_path = os.path.join(cur_dir, 'dataset/t5-large')
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)


    origin_vocab_len = len(tokenizer) # origin_len, using for adjust embeddings.
    # print('previous',origin_vocab_len) # debug
    # load new_tokenizer 
    # tokenizer, dataset = load_tokenized_dataset(model_name, tokenizer)
    # print(len(tokenizer)) # debug
    # sys.exit() # debug
    # resize model vocab:
    if model_name == 't5-large':
        model.resize_token_embeddings(len(tokenizer))
        pass
    elif model_name == 't5-base':
        # origin size: torch.Size([32128, 768])
        model.resize_token_embeddings(32129)
    
    
    # use w/o embedding padding?
    resize_model_embedding(origin_len=origin_vocab_len, new_tokenizer=tokenizer, model=model)

    # the training code can be used for multi times, need to config the dataset name

    # ckpt_path = os.path.join(cur_dir, 'model/alpaca_instruction/checkpoint_final.pkl')
    # check_point = torch.load(ckpt_path)['model_state_dict']
    # check_point = {k.replace('module.', ''):v for k,v in check_point.items()}
    # model.load_state_dict(check_point)

    # using half model
    model = model.half().cuda()
    
    training_args = {
        "output_dir": f'model/alpaca_instruction',
        "per_gpu_train_batch_size": 8,
        "per_gpu_eval_batch_size": 16,
        # "learning_rate": 1e-4,
        "num_train_epochs": 3,
    }

    # num_epochs = training_args['num_train_epochs']
    # train_bs = training_args['per_gpu_train_batch_size']
    valid_bs = training_args['per_gpu_eval_batch_size']

    # sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], shuffle=True)
    # train_iter = DataLoader(dataset['train'], batch_size=train_bs, sampler=sampler)
    # 
    valid_iter = DataLoader(dataset['validation'], batch_size=valid_bs, shuffle=False) # not use for now.
    
    # train(model, None, train_iter, valid_iter, num_epochs, training_args['output_dir'], local_rank)
    # train(model, optimizer, train_iter, valid_iter, num_epochs, training_args['output_dir'], local_rank)\

    output_path = 'infer_origin.json'

    decomposition_generation(tokenizer, model, valid_iter, output_path)




    

