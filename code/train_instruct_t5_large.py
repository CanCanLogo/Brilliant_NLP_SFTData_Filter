from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, T5Tokenizer, T5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR 
import torch.optim as optim
# from torch_optimizer import Adafactor
import deepspeed

import os
from tqdm import tqdm
# from musique_transformers_dataset import get_dataset
import sys
sys.path.append('./utils')
from utils.utils_instruction import load_tokenized_dataset

# AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
# print(os.environ)
os.environ["RANK"] = "1"

os.environ["LOCAL_RANK"] = "0"

os.environ["WORLD_SIZE"] = "1"


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

deepspeed.ops.op_builder.CPUAdamBuilder().load()

cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).cuda()
    return tokenizer, model

def train(model, optimizer, train_iter, valid_iter, epochs, output_dir, local_rank):
    total = len(train_iter) * epochs
    with tqdm(total=total) as pbar:
        for epoch in range(1, epochs+1):
            for step, batch_input in enumerate(train_iter):
                
                batch_input = {k:v.cuda() for k,v in batch_input.items()}

                cur_loss = model(**batch_input).loss

                model.backward(cur_loss)
                model.step()

                pbar.set_postfix(loss=f'{cur_loss.item()}')
                pbar.update(1)
                pass
                # if local_rank==0 and step != 0 and step % 1000 == 0:
                if epoch < 3:
                    '''
                        start eval    
                    '''
                    # valid_loss = []
                    # with torch.no_grad():
                    #     for _, batch_valid in enumerate(valid_iter):
                    #         batch_valid = {k:v.cuda() for k,v in batch_valid.items()}
                    #         loss = model(**batch_valid).loss
                    #         valid_loss.append(loss.item())
                    # print(f'valid loss: {torch.mean(torch.tensor(valid_loss, dtype=float))}')
                    # save model checkpoint:
                    save_pth = os.path.join(output_dir, f'checkpoint_{epoch}e.pkl')
                    '''
                        torch.save
                    '''
                    torch.save({
                        "model_state_dict" : model.state_dict(),
                    }, save_pth)
    # if local_rank==0:
    save_pth = os.path.join(output_dir, f'checkpoint_final.pkl')
    '''
        torch.save
    '''
    torch.save({
        "model_state_dict" : model.state_dict(),
    }, save_pth)

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

if __name__ == '__main__':
    # init distributed envs:
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # deepspeed.init_distributed(dist_backend='nccl', rank=local_rank, world_size=world_size)
    '''
    important there is rank
    '''
    torch.cuda.set_device(1)
    # 'cuda:'+os.environ['CUDA_VISIBLE_DEVICES']
    training_args = {
        "output_dir": f'model/alpaca_instruction_raw',
        "per_gpu_train_batch_size": 8,
        "per_gpu_eval_batch_size": 16,
        # "learning_rate": 1e-4,
        "num_train_epochs": 3,
    }

    ds_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 2,
        # "train_micro_batch_size_per_gpu": training_args['per_gpu_train_batch_size'],
        "communication_data_type": "fp32",
        'localhost': '1',

        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"}
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [
                0.8,
                0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000 }},
        # using bf16 instead of fp16
        'bf16': {
            "enabled": True
        },
        'enable_cuda_graph': True,
        'use_triton': True,
        # try to add
        "gradient_clipping": 1.0,
    }

    model_name = 't5-large'
    model_path = 'model/t5_decomposer' if model_name=='t5-base' else 'model/flan-t5-large'
    # tokenizer, model = load_model(model_path)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(cur_dir, model_path))
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(cur_dir, model_path)).cuda()

    origin_vocab_len = len(tokenizer) # origin_len, using for adjust embeddings.
    # print('previous',origin_vocab_len) # debug
    # load new_tokenizer 
    tokenizer, dataset = load_tokenized_dataset(model_name, tokenizer)
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

    # initialize deepspeed engine:
    model, optimizer, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config)

    num_epochs = training_args['num_train_epochs']
    train_bs = training_args['per_gpu_train_batch_size']
    valid_bs = training_args['per_gpu_eval_batch_size']

    sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], shuffle=True)
    train_iter = DataLoader(dataset['train'], batch_size=train_bs, sampler=sampler)
    # 
    valid_iter = DataLoader(dataset['validation'], batch_size=valid_bs, shuffle=False) # not use for now.
    
    # train(model, None, train_iter, valid_iter, num_epochs, training_args['output_dir'], local_rank)
    train(model, optimizer, train_iter, valid_iter, num_epochs, training_args['output_dir'], local_rank)

    
    # # 提取DeepSpeed配置中的优化器参数  
    # optimizer_params = {  
    #     'lr': 0.001,  
    #     'betas': (0.8, 0.999),  
    #     'eps': 1e-8,  
    #     'weight_decay': 3e-7  
    # }  
    
    # # 初始化优化器  
    # optimizer = optim.Adam(model.parameters(), **optimizer_params)  
    
    # # 自定义warmup学习率调度器  
    # def warmup_lr_lambda(step):  
    #     warmup_num_steps = 1000  
    #     if step < warmup_num_steps:  
    #         return step / warmup_num_steps  
    #     return 1.0  
    
    # # 初始化学习率调度器  
    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)  
    
    # # 注意：PyTorch没有直接的bf16支持（截至2023年），但你可以使用AMP（自动混合精度）  
    # # 或者，如果你使用的是PyTorch 1.10或更新版本，可以使用torch.cuda.amp.autocast  
    # # 在这里，我们不会设置bf16，但会提到如何使用AMP  
    
    # # 使用梯度裁剪  
    # for group in optimizer.param_groups:  
    #     group['clipnorm'] = 1.0  # 注意：这是梯度范数裁剪


    # optimizer = Adafactor(model.parameters(), lr=training_args['learning_rate'])



    

