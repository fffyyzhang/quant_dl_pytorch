#!/bin/bash

# 设置数据文件路径
DATA_PATH="data/stock_data.csv"

# 设置模型类型
MODEL_TYPE="LSTM"

# 设置输出目录
OUTPUT_DIR="example_output"

# 运行 Python 脚本并调用 run_complete_example 函数
python3 main.py --mode example --data "$DATA_PATH" --model-type "$MODEL_TYPE" --output "$OUTPUT_DIR" 