#!/usr/bin/env python3
"""
简单的模型选择示例

展示如何在训练过程中简单地选择不同的模型
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train.model_utils import create_model, get_available_models


def demo_model_creation():
    """演示模型创建"""
    print("=== 可用的模型类型 ===")
    available_models = get_available_models()
    print(f"支持的模型: {available_models}")
    
    print("\n=== 创建不同类型的模型 ===")
    
    # 模型参数
    model_params = {
        'input_size': 5,      # OHLCV
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 2,
        'dropout': 0.2
    }
    
    # 测试几种模型类型
    test_models = ['lstm', 'gru', 'cnn']
    
    for model_name in test_models:
        if model_name in available_models:
            try:
                model = create_model(model_name, **model_params)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"{model_name.upper():12} - {model.__class__.__name__:20} - 参数: {param_count:,}")
            except Exception as e:
                print(f"{model_name.upper():12} - 创建失败: {e}")


def demo_training_with_model_selection():
    """演示训练中的模型选择"""
    print("\n=== 训练中的模型选择示例 ===")
    
    # 假设我们有训练配置
    training_configs = [
        {'model_type': 'lstm', 'hidden_size': 64},
        {'model_type': 'gru', 'hidden_size': 64},
        {'model_type': 'cnn', 'hidden_size': 64, 'kernel_size': 3},
    ]
    
    print("可以尝试的训练配置:")
    for i, config in enumerate(training_configs, 1):
        print(f"{i}. 模型: {config['model_type']}, 配置: {config}")
    
    print("\n使用方法:")
    print("python train/train_etf.py --data_dir data/ --model_type lstm")
    print("python train/train_etf.py --data_dir data/ --model_type gru")
    print("python train/train_etf.py --data_dir data/ --model_type cnn")


def demo_command_line_usage():
    """演示命令行使用方法"""
    print("\n=== 命令行使用示例 ===")
    
    examples = [
        {
            'description': '使用LSTM模型训练',
            'command': 'python train/train_etf.py --data_dir processed_data/ --model_type lstm --hidden_size 64'
        },
        {
            'description': '使用GRU模型训练',
            'command': 'python train/train_etf.py --data_dir processed_data/ --model_type gru --hidden_size 128'
        },
        {
            'description': '使用CNN模型训练',
            'command': 'python train/train_etf.py --data_dir processed_data/ --model_type cnn --hidden_size 64'
        },
        {
            'description': '使用主程序训练',
            'command': 'python main.py --mode train --data data.csv --model-type lstm'
        }
    ]
    
    for example in examples:
        print(f"\n{example['description']}:")
        print(f"  {example['command']}")


def main():
    """主函数"""
    print("简单模型选择功能演示")
    print("=" * 50)
    
    # 演示模型创建
    demo_model_creation()
    
    # 演示训练中的模型选择
    demo_training_with_model_selection()
    
    # 演示命令行使用
    demo_command_line_usage()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n核心功能:")
    print("1. 只需要在命令行指定 --model_type 参数")
    print("2. 支持的模型类型:", get_available_models())
    print("3. 其他参数如 hidden_size, num_layers 等都可以自定义")


if __name__ == '__main__':
    main() 