#!/usr/bin/env python3
"""
基础使用示例

展示如何使用股价预测PyTorch模型包进行基本的训练和预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction_pytorch.data import create_loaders_from_file
from stock_prediction_pytorch.models import LSTMModel
from stock_prediction_pytorch.training import StockTrainer
from stock_prediction_pytorch.prediction import StockPredictor
from stock_prediction_pytorch.utils import Config, set_seed


def basic_training_example():
    """基础训练示例"""
    print("=== 基础训练示例 ===")
    
    # 设置随机种子
    set_seed(42)
    
    # 使用参考数据（需要将ref_codes中的数据复制到data目录）
    data_path = '/home/liyuan/proj_liy/quant_dl_pytorch/ref_codes/Stock-Prediction-Models/Good-year.csv'
    
    # 创建数据加载器
    train_loader, val_loader, preprocessor = create_loaders_from_file(
        data_path=data_path,
        target_column='Close',
        sequence_length=10,
        test_size=30,
        batch_size=32,
        normalize=True
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    
    # 创建LSTM模型
    model = LSTMModel(
        input_size=train_loader.dataset.get_feature_dim(),
        hidden_size=64,
        output_size=train_loader.dataset.get_target_dim(),
        num_layers=2,
        dropout=0.2
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = StockTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        patience=5
    )
    
    # 训练模型（使用较少轮数作为示例）
    print("开始训练...")
    history = trainer.train(num_epochs=20)
    
    # 保存模型
    trainer.save_model('basic_model.pth')
    
    print("训练完成！")
    return model, preprocessor


def basic_prediction_example(model, preprocessor):
    """基础预测示例"""
    print("\n=== 基础预测示例 ===")
    
    # 创建预测器
    predictor = StockPredictor(model=model, preprocessor=preprocessor)
    
    # 使用相同数据进行预测（实际使用中应该使用新数据）
    data_path = '../ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv'
    
    # 进行预测
    result = predictor.predict_from_file(
        data_path=data_path,
        output_path='basic_predictions.csv',
        future_steps=5
    )
    
    # 打印结果
    print("预测指标:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if result['future_predictions'] is not None:
        print(f"\n未来5天预测:")
        for i, pred in enumerate(result['future_predictions']):
            print(f"  第{i+1}天: {pred:.2f}")
    
    return result


def config_based_example():
    """基于配置的示例"""
    print("\n=== 基于配置的示例 ===")
    
    # 创建配置
    config = Config()
    config.model.model_type = 'LSTM'
    config.model.hidden_size = 128
    config.model.num_layers = 3
    config.model.dropout = 0.3
    
    config.training.learning_rate = 0.0005
    config.training.batch_size = 16
    config.training.num_epochs = 15
    config.training.patience = 7
    
    config.data.sequence_length = 15
    config.data.test_size = 40
    
    # 保存配置
    config.save('example_config.json')
    print("配置已保存到 example_config.json")
    
    # 加载配置
    loaded_config = Config.load('example_config.json')
    print("配置加载成功")
    
    # 打印配置信息
    loaded_config.print_config()
    
    return loaded_config


def main():
    """主函数"""
    try:
        # 检查数据文件是否存在
        #data_path = '../ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv'
        data_path = '/home/liyuan/proj_liy/quant_dl_pytorch/ref_codes/Stock-Prediction-Models/Good-year.csv'
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            print("请确保参考代码中的数据文件可用")
            return
        
        # 基础训练示例
        model, preprocessor = basic_training_example()
        
        # 基础预测示例
        result = basic_prediction_example(model, preprocessor)
        
        # 配置示例
        config = config_based_example()
        
        print("\n=== 示例完成 ===")
        print("生成的文件:")
        print("- basic_model.pth: 训练好的模型")
        print("- basic_predictions.csv: 预测结果")
        print("- example_config.json: 配置文件")
        
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
        print("请检查依赖是否正确安装")


if __name__ == '__main__':
    main() 