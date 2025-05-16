from tqdm import tqdm
import math
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import io
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


# 读取json文件
def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


device = "cuda" if torch.cuda.is_available() else "cpu"

'''----------------------参数与地址-----------------------------'''
file_in = r"D:\new_program\pythonProject\pytorchUse\NLPpro1\NLPNewpro2\alpaca_data.json"
high_quality_file = "./high_quality.json"
# 需要调整的参数
threshold = 3
'''---------------------------------------------------'''

# 读取数据
input_list = jload(file_in)
print('number of input file', len(input_list))

# 调用reward-model-deberta-v3-large-v2模型
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_name = r"D:\new_program\pythonProject\pytorchUse\NLPpro1\NLPNewpro2\model\reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(
    reward_name)
rank_model = rank_model.to(device)

# 打分测试
question, answer = "Explain nuclear fusion like I am five", \
    "Nuclear fusion is the process by which two or more protons and neutrons\n" \
    " combine to form a single nucleus. It is a very important process in the universe,\n " \
    "as it is the source of energy for stars and galaxies. Nuclear fusion is also a key \n" \
    "process in the production of energy for nuclear power plants."

inputs = tokenizer(question, answer, return_tensors='pt').to(device)
#
score = rank_model(**inputs).logits[0].detach()
print(float(score))

# 读取指令并打分
result_list = []
num_dict = {}
result_json = []

for element in tqdm(input_list):
    # 将指令，输入与输出区分开
    instruction = element['instruction']
    _input = ''
    if 'input' in element.keys():
        _input = element['input']
    _output = element['output']
    question = ''
    if _input == '':
        question = instruction
    else:
        question = instruction + '\n' + _input  # 指令与输入是问题，对应user

    answer = _output

    try:
        inputs = tokenizer(question, answer, return_tensors='pt').to(device)
        # .to(device)
        score = rank_model(**inputs).logits[0].detach()
    except:
        # print(instruction)
        # print(_output)
        # print(element)
        # print(question)
        # print(answer)
        # print()
        continue
    # 存入指令输入输出与第一个模型的打分
    final_result = {'instruction': instruction, 'input': _input, 'output': _output, 'reward_score': float(score)}

    # 对得分向上取整
    upper_num = math.ceil(final_result['reward_score'])
    # 对得分向下取整
    lower_num = math.floor(final_result['reward_score'])
    # 统计每个得分段的数据数
    num_dict[(lower_num, upper_num)] = num_dict.get((lower_num, upper_num), 0) + 1
    # 保存高得分数据
    # if float(final_result['reward_score']) > threshold:
    result_json.append(final_result)
    # 统计每个得分段的数据数
    num_dict[(lower_num, upper_num)] = num_dict.get((lower_num, upper_num), 0) + 1


print('num of good case : ', len(result_json))
print('The percent of each score interval:')
all_num = len(num_dict)
for k, v in num_dict.items():
    print(str(k) + '  :  ' + str(v) + '  ' + str(float(v) / all_num))


# 保存高得分数据
jdump(result_json, high_quality_file)
