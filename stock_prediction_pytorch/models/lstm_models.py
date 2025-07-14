"""
LSTM系列模型

基于PyTorch实现的LSTM股价预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_model import StockLSTMBase


class LSTMModel(StockLSTMBase):
    """基础LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: LSTM层数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, **kwargs)
        
    def _build_model(self):
        """构建模型架构"""
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 输出层
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(lstm_output_size, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h0, c0)
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output


class BiLSTMModel(StockLSTMBase):
    """双向LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化双向LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: LSTM层数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        bidirectional=True, **kwargs)
        
    def _build_model(self):
        """构建双向LSTM模型架构"""
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层（双向LSTM输出维度是hidden_size * 2）
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h0, c0)
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 双向LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output


class MultiLSTMModel(StockLSTMBase):
    """多层LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 **kwargs):
        """
        初始化多层LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: LSTM层数
            dropout: dropout概率
            bidirectional: 是否双向
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        bidirectional=bidirectional, **kwargs)
        
    def _build_model(self):
        """构建多层LSTM模型架构"""
        # 多层LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 输出层
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h0, c0)
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # 多层LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(lstm_out)
        
        return output


class LSTM2PathModel(StockLSTMBase):
    """2路径LSTM模型（参考原始代码）"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化2路径LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: LSTM层数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, **kwargs)
        
    def _build_model(self):
        """构建2路径LSTM模型架构"""
        # 第一路径LSTM
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 第二路径LSTM
        self.lstm2 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 融合层
        self.fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # 输出层
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h0, c0)
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden1 = self.init_hidden(batch_size, x.device)
            hidden2 = self.init_hidden(batch_size, x.device)
        else:
            hidden1, hidden2 = hidden
        
        # 第一路径LSTM
        lstm_out1, _ = self.lstm1(x, hidden1)
        lstm_out1 = lstm_out1[:, -1, :]
        
        # 第二路径LSTM
        lstm_out2, _ = self.lstm2(x, hidden2)
        lstm_out2 = lstm_out2[:, -1, :]
        
        # 融合两路径特征
        combined = torch.cat([lstm_out1, lstm_out2], dim=1)
        fused = self.fusion(combined)
        fused = F.relu(fused)
        
        # Dropout
        fused = self.dropout_layer(fused)
        
        # 输出层
        output = self.fc(fused)
        
        return output


class LSTMAttentionModel(StockLSTMBase):
    """带注意力机制的LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 attention_size: int = 64,
                 **kwargs):
        """
        初始化带注意力的LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: LSTM层数
            dropout: dropout概率
            attention_size: 注意力层大小
        """
        self.attention_size = attention_size
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        attention_size=attention_size, **kwargs)
        
    def _build_model(self):
        """构建带注意力的LSTM模型架构"""
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 注意力层
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        
        # 输出层
        self.fc = nn.Linear(lstm_output_size, self.output_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 隐藏状态 (h0, c0)
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)  # (batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 应用注意力权重
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        # Dropout
        attended_output = self.dropout_layer(attended_output)
        
        # 输出层
        output = self.fc(attended_output)
        
        return output 