#!/usr/bin/env python3
"""
ETF模型训练脚本

读取etf_simple_processor的输出数据，训练深度学习模型预测ETF价格
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_models import LSTMModel
from train.trainer import StockTrainer
from models import create_model as create_model_simple, get_available_models


def load_etf_data(data_dir: str, etf_code: Optional[str] = None):
    """
    加载ETF数据
    
    Args:
        data_dir: 数据目录路径
        etf_code: 指定的ETF代码，None则自动选择第一个
    
    Returns:
        train_loader, val_loader, data_info
    """
    # 加载处理结果信息
    info_path = os.path.join(data_dir, 'process_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"处理结果文件不存在: {info_path}")
    
    with open(info_path, 'r', encoding='utf8') as f:
        process_info = json.load(f)
    
    # 选择ETF
    if etf_code is None:
        # 默认合并全部ETF的训练和测试数据
        all_train_X, all_train_y, all_test_X, all_test_y = [], [], [], []
        for code, info in process_info.items():
            with open(info['train_path'], 'rb') as f:
                d = pickle.load(f)
                all_train_X.append(d['X'])
                all_train_y.append(d['y'])
            with open(info['test_path'], 'rb') as f:
                d = pickle.load(f)
                all_test_X.append(d['X'])
                all_test_y.append(d['y'])
        train_X = torch.FloatTensor(np.concatenate(all_train_X, axis=0))
        train_y = torch.FloatTensor(np.concatenate(all_train_y, axis=0))
        test_X = torch.FloatTensor(np.concatenate(all_test_X, axis=0))
        test_y = torch.FloatTensor(np.concatenate(all_test_y, axis=0))
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        print(f"已合并全部ETF数据: 训练样本={len(train_dataset)}, 测试样本={len(test_dataset)}")
        return train_dataset, test_dataset, {
            'etf_code': 'ALL',
            'input_size': train_X.shape[2],
            'sequence_length': train_X.shape[1],
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'timestamps_train': [],
            'timestamps_test': []
        }
    elif etf_code not in process_info:
        available_etfs = list(process_info.keys())
        raise ValueError(f"ETF {etf_code} 不存在，可用的ETF: {available_etfs}")
    
    etf_info = process_info[etf_code]
    print(f"加载ETF {etf_code} 数据...")
    print(f"训练样本: {etf_info['train_samples']}")
    print(f"测试样本: {etf_info['test_samples']}")
    
    # 加载训练数据
    with open(etf_info['train_path'], 'rb') as f:
        train_data = pickle.load(f)
    
    # 加载测试数据
    with open(etf_info['test_path'], 'rb') as f:
        test_data = pickle.load(f)
    
    # 创建PyTorch数据集
    train_X = torch.FloatTensor(train_data['X'])
    train_y = torch.FloatTensor(train_data['y'])
    test_X = torch.FloatTensor(test_data['X'])
    test_y = torch.FloatTensor(test_data['y'])
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    print(f"数据形状: X={train_X.shape}, y={train_y.shape}")
    
    return train_dataset, test_dataset, {
        'etf_code': etf_code,
        'input_size': train_X.shape[2],  # OHLCV = 5
        'sequence_length': train_X.shape[1],
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'timestamps_train': train_data.get('timestamps', []),
        'timestamps_test': test_data.get('timestamps', [])
    }


def create_model(input_size: int, config: dict):
    """
    创建模型 - 支持多种模型类型
    
    Args:
        input_size: 输入特征数
        config: 模型配置
    
    Returns:
        模型实例
    """
    model_type = config.get('model_type', 'lstm')
    
    # 使用简化的模型创建工具
    if model_type.lower() in get_available_models():
        return create_model_simple(
            model_name=model_type,
            input_size=input_size,
            hidden_size=config.get('hidden_size', 64),
            output_size=1,  # 预测一个值
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        )
    else:
        # 保持向后兼容，支持原来的lstm
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=config.get('hidden_size', 64),
                output_size=1,
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )
            return model
        else:
            available = get_available_models()
            raise ValueError(f"不支持的模型类型: {model_type}，可用模型: {available}")


def main():
    parser = argparse.ArgumentParser(description='ETF模型训练')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径（etf_simple_processor的输出）')
    parser.add_argument('--etf_code', type=str, default=None,
                       help='指定训练的ETF代码，默认自动选择第一个')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=get_available_models() + ['lstm'], help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备类型 (cpu/cuda/auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    
    # 新增模型特定参数
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Transformer注意力头数')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='CNN卷积核大小')
    
    args = parser.parse_args()
    
    # 加载数据
    print("=" * 50)
    print("加载数据...")
    train_dataset, test_dataset, data_info = load_etf_data(args.data_dir, args.etf_code)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    print("=" * 50)
    print("创建模型...")
    model_config = {
        'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'num_heads': args.num_heads,      # Transformer参数
        'kernel_size': args.kernel_size   # CNN参数
    }
    
    model = create_model(data_info['input_size'], model_config)
    print(f"模型: {model.__class__.__name__}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 创建训练器
    print("=" * 50)
    print("开始训练...")
    
    trainer = StockTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        patience=args.patience,
        save_dir=args.save_dir
    )
    
    # 训练模型
    history = trainer.train(
        num_epochs=args.epochs,
        save_best_only=True,
        verbose=True
    )
    
    # 保存训练信息
    result_info = {
        'etf_code': data_info['etf_code'],
        'model_config': model_config,
        'training_config': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'device': device
        },
        'data_info': {
            'input_size': data_info['input_size'],
            'sequence_length': data_info['sequence_length'],
            'train_samples': data_info['train_samples'],
            'test_samples': data_info['test_samples']
        },
        'training_history': history
    }
    
    # 保存结果信息
    result_path = os.path.join(args.save_dir, f"{data_info['etf_code']}_training_result.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(result_info, f, ensure_ascii=False, indent=2)
    
    print("=" * 50)
    print("训练完成!")
    print(f"最佳模型保存在: {args.save_dir}")
    print(f"训练结果保存在: {result_path}")
    
    # 简单评估
    print("=" * 50)
    print("最终评估:")
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"测试集平均损失: {avg_loss:.6f}")
    print(f"测试集RMSE: {np.sqrt(avg_loss):.6f}")


if __name__ == "__main__":
    main() 