"""
基础股价预测模型类

为所有股价预测模型提供统一的接口和通用功能
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseStockModel(nn.Module, ABC):
    """基础股价预测模型抽象类"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化基础模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: 层数
            dropout: dropout概率
            **kwargs: 其他模型特定参数
        """
        super(BaseStockModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 模型配置
        self.model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'dropout': dropout,
            **kwargs
        }
        
        # 初始化模型组件
        self._build_model()
        
    @abstractmethod
    def _build_model(self):
        """构建模型架构 - 子类必须实现"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播 - 子类必须实现"""
        pass
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            初始隐藏状态
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.__class__.__name__,
            'config': self.model_config,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'model_class': self.__class__.__name__
        }, path)
    
    @classmethod
    def load_model(cls, path: str, **kwargs):
        """
        加载模型
        
        Args:
            path: 模型路径
            **kwargs: 额外参数
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['model_config']
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict(self, x: torch.Tensor, device: torch.device = None) -> np.ndarray:
        """
        预测
        
        Args:
            x: 输入数据
            device: 设备
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if device is not None:
                x = x.to(device)
                self.to(device)
            
            output = self.forward(x)
            return output.cpu().numpy()
    
    def predict_sequence(self, 
                        x: torch.Tensor, 
                        future_steps: int,
                        device: torch.device = None) -> np.ndarray:
        """
        序列预测（递归预测多步）
        
        Args:
            x: 输入序列
            future_steps: 预测步数
            device: 设备
            
        Returns:
            预测序列
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            if device is not None:
                x = x.to(device)
                self.to(device)
            
            current_input = x.clone()
            
            for _ in range(future_steps):
                # 预测下一步
                pred = self.forward(current_input)
                predictions.append(pred.cpu().numpy())
                
                # 更新输入序列（滑动窗口）
                if len(current_input.shape) == 3:  # (batch, seq_len, features)
                    # 移除第一个时间步，添加预测结果
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        pred.unsqueeze(1)
                    ], dim=1)
                else:  # (seq_len, features)
                    current_input = torch.cat([
                        current_input[1:, :],
                        pred.unsqueeze(0)
                    ], dim=0)
        
        return np.array(predictions)
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算预测准确率（参考原始代码）
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            准确率百分比
        """
        y_true = np.array(y_true) + 1
        y_pred = np.array(y_pred) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
        return percentage * 100
    
    def get_model_summary(self) -> str:
        """获取模型摘要"""
        info = self.get_model_info()
        summary = f"""
        Model: {info['model_name']}
        Input Size: {info['config']['input_size']}
        Hidden Size: {info['config']['hidden_size']}
        Output Size: {info['config']['output_size']}
        Layers: {info['config']['num_layers']}
        Dropout: {info['config']['dropout']}
        Total Parameters: {info['total_params']:,}
        Trainable Parameters: {info['trainable_params']:,}
        """
        return summary


class StockRNNBase(BaseStockModel):
    """RNN类型模型的基础类"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 **kwargs):
        """
        初始化RNN基础模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: 层数
            dropout: dropout概率
            bidirectional: 是否双向
            **kwargs: 其他参数
        """
        self.bidirectional = bidirectional
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        bidirectional=bidirectional, **kwargs)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        初始化RNN隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            初始隐藏状态
        """
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * num_directions, batch_size, 
                          self.hidden_size, device=device)


class StockLSTMBase(StockRNNBase):
    """LSTM类型模型的基础类"""
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, 
                        self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, 
                        self.hidden_size, device=device)
        return h0, c0


class StockTransformerBase(BaseStockModel):
    """Transformer类型模型的基础类"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 num_heads: int = 8,
                 **kwargs):
        """
        初始化Transformer基础模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: 层数
            dropout: dropout概率
            num_heads: 注意力头数
            **kwargs: 其他参数
        """
        self.num_heads = num_heads
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        num_heads=num_heads, **kwargs)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Transformer不需要隐藏状态"""
        return None 