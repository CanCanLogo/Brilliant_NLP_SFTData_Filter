import json  
import pandas as pd  
from sklearn.model_selection import train_test_split  
import random
  
# 读取JSON文件  
# with open('seed-instructions.json', 'r', encoding='utf-8') as f:  
#     data = json.load(f)  

# with open('high_quality_3.json', 'r', encoding='utf-8') as f:  
#     data = json.load(f) 

with open('alpaca_data.json', 'r', encoding='utf-8') as f:  
    data = json.load(f) 
  
# # 假设JSON数据是一个列表，其中每个元素是一个字典  
# # 将其转换为pandas DataFrame  
# df = pd.DataFrame(data)  
  
# # 合并两个字段为一个新字段  
# df['input_context'] = df['instruction'] + df['input']  

# # 修改字段名  
# df = df.rename(columns={'input_context': 'input_context', 'output': 'output_question'})  

# columns_to_drop = ['instruction', 'reward_score', 'input']
# df.drop(columns=columns_to_drop, axis=1, inplace=True)  

# df['input_context'] = df['input_context'].str.replace('\n', '', regex=True)  
# df['output_question'] = df['output_question'].str.replace('\n', '', regex=True) 
  
# # 拆分数据集为训练集和验证集  
# # 假设我们想要80%的数据作为训练集，剩下的20%作为验证集  
# train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)  
  
# # 将训练集和验证集写入CSV文件  
# train_df.to_csv('train.csv', index=False, encoding='utf-8', sep='\t')  
# valid_df.to_csv('valid.csv', index=False, encoding='utf-8', sep='\t')


processed_data = []  
for item in data:  
    # 去掉 field4  
    if 'reward_score' in item:  
        del item['reward_score']  
      
    # 合并 field2 和 field3 到 combinedField  
    combined_value = f"{item['instruction']} {item['input']}"  # 或者你可以使用其他合并方式  
    item['input_context'] = combined_value  
      
    # 如果需要，可以在这步之前或之后重命名 field2  
    # del item['field2']  # 如果你不再需要 field2  
      
    # 重命名 field1 为 newName1  
    item['output_question'] = item.pop('output')  
      
    # 如果不再需要 field2 和 field3，可以在这里删除它们  
    del item['instruction']  
    del item['input']  
      
    processed_data.append(item)  
  
# 步骤3：写入新的JSON文件  
with open('processed_data.json', 'w') as f:  
    json.dump(processed_data, f, indent=4)



with open('processed_data.json', 'r', encoding='utf-8') as f:  
    data = json.load(f)  
 
  # 步骤2：随机分割数据  
# 假设我们想要80%的数据用于训练，20%的数据用于测试  
train_size = int(0.8 * len(data))  
# train_data, test_data = data[:train_size], data[train_size:]  
  
# 为了确保分割是随机的，你可以在分割之前先打乱数据  
random.shuffle(data)  
train_data, test_data = data[:train_size], data[train_size:]
  
# 步骤3：写入新的JSON文件  
with open('train_raw.json', 'w') as f:  
    json.dump(train_data, f, indent=4)  
  
with open('test_raw.json', 'w') as f:  
    json.dump(test_data, f, indent=4)