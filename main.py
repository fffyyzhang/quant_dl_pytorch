#!/usr/bin/env python3
"""
股价预测PyTorch模型包主程序

使用示例和完整的训练预测流程
"""

import os
import argparse
import json
from typing import Dict, Any, Optional
import pandas as pd

# 导入我们的包
from stock_prediction_pytorch.data import StockPreprocessor, create_loaders_from_file
from stock_prediction_pytorch.models import (
    LSTMModel, BiLSTMModel, GRUModel, TransformerModel, 
    CNNSeq2SeqModel, DilatedCNNModel
)
from stock_prediction_pytorch.training import StockTrainer
from stock_prediction_pytorch.prediction import StockPredictor
from stock_prediction_pytorch.utils import Config, set_seed, get_device, print_system_info


def create_model(config: Config):
    """
    根据配置创建模型 - 使用简化的模型创建方式
    
    Args:
        config: 配置对象
        
    Returns:
        模型实例
    """
    # 导入简化的模型创建工具
    from models import create_model as create_model_simple, get_available_models
    
    model_type = config.model.model_type.lower()
    model_params = config.get_model_params()
    
    # 移除model_type参数
    model_params.pop('model_type', None)
    
    # 使用简化的创建方式
    try:
        return create_model_simple(model_type, **model_params)
    except ValueError:
        # 如果简化方式失败，尝试原来的方式（向后兼容）
        model_type = config.model.model_type.upper()
        
        if model_type == 'LSTM':
            return LSTMModel(**model_params)
        elif model_type == 'BILSTM':
            return BiLSTMModel(**model_params)
        elif model_type == 'GRU':
            return GRUModel(**model_params)
        elif model_type == 'TRANSFORMER':
            return TransformerModel(**model_params)
        elif model_type == 'CNN':
            return CNNSeq2SeqModel(**model_params)
        elif model_type == 'DILATEDCNN':
            return DilatedCNNModel(**model_params)
        else:
            available = get_available_models()
            raise ValueError(f"不支持的模型类型: {config.model.model_type}，可用模型: {available}")


def train_model(config: Config, data_path: str, save_dir: str = 'checkpoints'):
    """
    训练模型
    
    Args:
        config: 配置对象
        data_path: 数据文件路径
        save_dir: 模型保存目录
        
    Returns:
        训练历史
    """
    print("=== 开始训练模型 ===")
    
    # 设置随机种子
    set_seed(42)
    
    # 获取设备
    device = get_device()
    
    # 创建数据加载器
    print("正在准备数据...")
    train_loader, val_loader, preprocessor = create_loaders_from_file(
        data_path=data_path,
        target_column=config.data.target_column,
        feature_columns=config.data.feature_columns,
        sequence_length=config.data.sequence_length,
        test_size=config.data.test_size,
        batch_size=config.training.batch_size,
        normalize=config.data.normalize
    )
    
    # 更新配置中的输入和输出大小
    config.model.input_size = train_loader.dataset.get_feature_dim()
    config.model.output_size = train_loader.dataset.get_target_dim()
    
    print(f"数据准备完成，训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    
    # 创建模型
    print("正在创建模型...")
    model = create_model(config)
    print(f"模型创建完成: {model.__class__.__name__}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = StockTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=config.training.learning_rate,
        optimizer=config.training.optimizer,
        loss_function=config.training.loss_function,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        save_dir=save_dir
    )
    
    # 开始训练
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        scheduler=config.training.scheduler,
        scheduler_params=config.training.scheduler_params,
        save_best_only=config.training.save_best_only
    )
    
    # 保存配置和历史
    config.save(os.path.join(save_dir, 'config.json'))
    trainer.save_training_history()
    
    # 打印训练摘要
    summary = trainer.get_training_summary()
    print("\n=== 训练摘要 ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("================")
    
    return history, model, preprocessor


def predict_prices(model_path: str, 
                  data_path: str, 
                  config_path: Optional[str] = None,
                  output_dir: str = 'predictions',
                  future_steps: int = 30):
    """
    使用训练好的模型进行预测
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径
        config_path: 配置文件路径
        output_dir: 输出目录
        future_steps: 未来预测步数
    """
    print("=== 开始价格预测 ===")
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        config = Config.load(config_path)
    else:
        print("未找到配置文件，使用默认配置")
        config = Config()
    
    # 创建模型
    model = create_model(config)
    
    # 加载模型权重
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建预处理器
    preprocessor = StockPreprocessor(
        target_column=config.data.target_column,
        feature_columns=config.data.feature_columns,
        sequence_length=config.data.sequence_length,
        test_size=config.data.test_size,
        normalize=config.data.normalize
    )
    
    # 创建预测器
    predictor = StockPredictor(model=model, preprocessor=preprocessor)
    
    # 进行预测
    result = predictor.predict_from_file(
        data_path=data_path,
        output_path=os.path.join(output_dir, 'predictions.csv'),
        future_steps=future_steps
    )
    
    # 打印结果
    print("\n=== 预测结果 ===")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if result['future_predictions'] is not None:
        print(f"\n未来{future_steps}天预测:")
        for i, pred in enumerate(result['future_predictions']):
            print(f"第{i+1}天: {pred:.2f}")
    
    print("================")
    
    return result


def run_complete_example(data_path: str, 
                        model_type: str = 'LSTM',
                        output_dir: str = 'example_output'):
    """
    运行完整示例
    
    Args:
        data_path: 数据文件路径
        model_type: 模型类型
        output_dir: 输出目录
    """
    print(f"=== 运行完整示例: {model_type} ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置
    config = Config()
    config.model.model_type = model_type
    config.training.num_epochs = 50  # 示例用较少轮数
    config.training.batch_size = 32
    config.data.sequence_length = 10
    config.data.test_size = 30
    
    print("配置信息:")
    config.print_config()
    
    # 训练模型
    save_dir = os.path.join(output_dir, 'checkpoints')
    history, model, preprocessor = train_model(config, data_path, save_dir)
    
    # 找到最佳模型文件
    model_files = [f for f in os.listdir(save_dir) if f.startswith('best_model')]
    if model_files:
        best_model_path = os.path.join(save_dir, model_files[0])
        config_path = os.path.join(save_dir, 'config.json')
        
        # 进行预测
        pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        result = predict_prices(
            model_path=best_model_path,
            data_path=data_path,
            config_path=config_path,
            output_dir=pred_dir,
            future_steps=10
        )
        
        print(f"\n示例完成！结果保存在: {output_dir}")
        
        return result
    else:
        print("未找到训练好的模型文件")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股价预测PyTorch模型包')
    parser.add_argument('--mode', choices=['train', 'predict', 'example'], 
                       required=True, help='运行模式')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--model', help='模型文件路径（预测模式需要）')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--model-type', default='LSTM', 
                       choices=['LSTM', 'BiLSTM', 'GRU', 'Transformer', 'CNN', 'DilatedCNN'],
                       help='模型类型')
    parser.add_argument('--future-steps', type=int, default=30, help='未来预测步数')
    
    args = parser.parse_args()
    
    # 打印系统信息
    print_system_info()
    
    try:
        if args.mode == 'train':
            # 训练模式
            if args.config:
                config = Config.load(args.config)
            else:
                config = Config()
                config.model.model_type = args.model_type
            
            train_model(config, args.data, args.output)
            
        elif args.mode == 'predict':
            # 预测模式
            if not args.model:
                raise ValueError("预测模式需要指定模型文件路径 (--model)")
            
            predict_prices(
                model_path=args.model,
                data_path=args.data,
                config_path=args.config,
                output_dir=args.output,
                future_steps=args.future_steps
            )
            
        elif args.mode == 'example':
            # 示例模式
            run_complete_example(
                data_path=args.data,
                model_type=args.model_type,
                output_dir=args.output
            )
            
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 