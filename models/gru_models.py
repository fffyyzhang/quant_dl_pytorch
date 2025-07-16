"""
GRU系列模型

基于PyTorch实现的GRU股价预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_model import StockRNNBase


class GRUModel(StockRNNBase):
    """基础GRU模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: GRU层数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, **kwargs)
        
    def _build_model(self):
        """构建模型架构"""
        # GRU层
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 输出层
        gru_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(gru_output_size, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # GRU前向传播
        gru_out, hidden = self.gru(x, hidden)
        
        # 取最后一个时间步的输出
        gru_out = gru_out[:, -1, :]
        
        # Dropout
        gru_out = self.dropout_layer(gru_out)
        
        # 全连接层
        output = self.fc(gru_out)
        
        return output


class BiGRUModel(StockRNNBase):
    """双向GRU模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化双向GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: GRU层数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        bidirectional=True, **kwargs)
        
    def _build_model(self):
        """构建双向GRU模型架构"""
        # 双向GRU层
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层（双向GRU输出维度是hidden_size * 2）
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 双向GRU前向传播
        gru_out, hidden = self.gru(x, hidden)
        
        # 取最后一个时间步的输出
        gru_out = gru_out[:, -1, :]
        
        # Dropout
        gru_out = self.dropout_layer(gru_out)
        
        # 全连接层
        output = self.fc(gru_out)
        
        return output


class MultiGRUModel(StockRNNBase):
    """多层GRU模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 **kwargs):
        """
        初始化多层GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: GRU层数
            dropout: dropout概率
            bidirectional: 是否双向
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        bidirectional=bidirectional, **kwargs)
        
    def _build_model(self):
        """构建多层GRU模型架构"""
        # 多层GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 输出层
        gru_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 多层GRU前向传播
        gru_out, hidden = self.gru(x, hidden)
        
        # 取最后一个时间步的输出
        gru_out = gru_out[:, -1, :]
        
        # 全连接层
        output = self.fc(gru_out)
        
        return output 