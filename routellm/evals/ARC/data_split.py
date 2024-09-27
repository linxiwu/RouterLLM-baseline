import json

# 假设原始文件名为 'data.json'
input_file = '/home/wenhao/Project/wlx/RouteLLM/routellm/evals/ARC/ARC.json'
output_train_file_json = 'math_train_data.json'
output_validation_file_json = 'math_validation_data.json'
output_train_file_jsonl = 'math_train_data.jsonl'
output_validation_file_jsonl = 'math_validation_data.jsonl'

# 读取JSON文件
with open(input_file, 'r') as file:
    data = json.load(file)

# 划分训练集和验证集
train_ratio = 0.75
train_size = int(len(data) * train_ratio)
train_data = data[:train_size]
validation_data = data[train_size:]

# 写入训练集和验证集到新的JSON文件
with open(output_train_file_json, 'w') as file:
    json.dump(train_data, file, indent=4)

with open(output_validation_file_json, 'w') as file:
    json.dump(validation_data, file, indent=4)

print(f"训练集已保存到 {output_train_file_json}")
print(f"验证集已保存到 {output_validation_file_json}")

# 将训练集转换为JSONL文件
with open(output_train_file_jsonl, 'w') as file:
    for item in train_data:
        file.write(json.dumps(item) + '\n')  # 将每个JSON对象作为一行写入

# 将验证集转换为JSONL文件
with open(output_validation_file_jsonl, 'w') as file:
    for item in validation_data:
        file.write(json.dumps(item) + '\n')

print(f"训练集已保存为 JSONL 文件: {output_train_file_jsonl}")
print(f"验证集已保存为 JSONL 文件: {output_validation_file_jsonl}")
