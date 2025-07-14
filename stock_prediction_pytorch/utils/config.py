"""
配置系统

管理模型超参数和训练参数的配置系统
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = 'LSTM'
    input_size: int = 1
    hidden_size: int = 128
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # 特定模型参数
    attention_size: int = 64
    kernel_size: int = 3
    num_heads: int = 8
    latent_size: int = 64


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = 'Adam'
    loss_function: str = 'MSE'
    
    # 早停配置
    patience: int = 10
    min_delta: float = 1e-6
    
    # 学习率调度
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    
    # 设备配置
    device: str = 'auto'
    num_workers: int = 0
    pin_memory: bool = True
    
    # 保存配置
    save_dir: str = 'checkpoints'
    save_best_only: bool = True
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class DataConfig:
    """数据配置"""
    target_column: str = 'Close'
    feature_columns: Optional[list] = None
    sequence_length: int = 5
    test_size: int = 30
    normalize: bool = True
    
    # 数据路径
    data_path: str = ''
    train_data_path: str = ''
    test_data_path: str = ''
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [self.target_column]


@dataclass
class PredictionConfig:
    """预测配置"""
    future_steps: int = 1
    method: str = 'recursive'  # 'recursive' or 'direct'
    confidence_intervals: bool = False
    n_samples: int = 100
    
    # 输出配置
    output_dir: str = 'predictions'
    save_predictions: bool = True
    save_plots: bool = True


class Config:
    """主配置类"""
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 prediction_config: Optional[PredictionConfig] = None):
        """
        初始化配置
        
        Args:
            model_config: 模型配置
            training_config: 训练配置
            data_config: 数据配置
            prediction_config: 预测配置
        """
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.prediction = prediction_config or PredictionConfig()
    
    def save(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 保存路径
        """
        config_dict = self.to_dict()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("不支持的文件格式，请使用 .json 或 .yaml")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        从文件加载配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            Config对象
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("不支持的文件格式，请使用 .json 或 .yaml")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config对象
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        prediction_config = PredictionConfig(**config_dict.get('prediction', {}))
        
        return cls(model_config, training_config, data_config, prediction_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'prediction': asdict(self.prediction)
        }
    
    def update(self, **kwargs):
        """
        更新配置参数
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套参数更新，如 'model.hidden_size'
                parts = key.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if hasattr(self, section):
                        section_config = getattr(self, section)
                        if hasattr(section_config, param):
                            setattr(section_config, param, value)
            else:
                # 直接设置参数
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        return asdict(self.model)
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        获取训练参数
        
        Returns:
            训练参数字典
        """
        return asdict(self.training)
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            是否有效
        """
        # 验证模型配置
        if self.model.input_size <= 0:
            raise ValueError("input_size必须大于0")
        
        if self.model.hidden_size <= 0:
            raise ValueError("hidden_size必须大于0")
        
        if self.model.output_size <= 0:
            raise ValueError("output_size必须大于0")
        
        if not 0 <= self.model.dropout < 1:
            raise ValueError("dropout必须在[0, 1)范围内")
        
        # 验证训练配置
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate必须大于0")
        
        if self.training.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs必须大于0")
        
        # 验证数据配置
        if self.data.sequence_length <= 0:
            raise ValueError("sequence_length必须大于0")
        
        if self.data.test_size <= 0:
            raise ValueError("test_size必须大于0")
        
        return True
    
    def print_config(self):
        """打印配置信息"""
        print("=== 配置信息 ===")
        print(f"模型类型: {self.model.model_type}")
        print(f"输入大小: {self.model.input_size}")
        print(f"隐藏大小: {self.model.hidden_size}")
        print(f"输出大小: {self.model.output_size}")
        print(f"层数: {self.model.num_layers}")
        print(f"Dropout: {self.model.dropout}")
        print(f"双向: {self.model.bidirectional}")
        print()
        print(f"学习率: {self.training.learning_rate}")
        print(f"批次大小: {self.training.batch_size}")
        print(f"训练轮数: {self.training.num_epochs}")
        print(f"优化器: {self.training.optimizer}")
        print(f"损失函数: {self.training.loss_function}")
        print()
        print(f"序列长度: {self.data.sequence_length}")
        print(f"测试大小: {self.data.test_size}")
        print(f"归一化: {self.data.normalize}")
        print(f"目标列: {self.data.target_column}")
        print("================")


def create_default_config() -> Config:
    """
    创建默认配置
    
    Returns:
        默认配置对象
    """
    return Config()


def load_config_from_file(filepath: str) -> Config:
    """
    从文件加载配置（便捷函数）
    
    Args:
        filepath: 配置文件路径
        
    Returns:
        Config对象
    """
    return Config.load(filepath)


def save_config_to_file(config: Config, filepath: str):
    """
    保存配置到文件（便捷函数）
    
    Args:
        config: 配置对象
        filepath: 保存路径
    """
    config.save(filepath)


# 预定义配置模板
LSTM_CONFIG = {
    'model': {
        'model_type': 'LSTM',
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': False
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100,
        'optimizer': 'Adam',
        'loss_function': 'MSE',
        'patience': 10
    }
}

TRANSFORMER_CONFIG = {
    'model': {
        'model_type': 'Transformer',
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1
    },
    'training': {
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 200,
        'optimizer': 'Adam',
        'loss_function': 'MSE',
        'patience': 15
    }
}

CNN_CONFIG = {
    'model': {
        'model_type': 'CNN',
        'hidden_size': 64,
        'num_layers': 3,
        'kernel_size': 3,
        'dropout': 0.2
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 150,
        'optimizer': 'Adam',
        'loss_function': 'MSE',
        'patience': 12
    }
} 