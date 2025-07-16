"""
CNN系列模型

基于PyTorch实现的CNN股价预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_model import BaseStockModel


class CNNSeq2SeqModel(BaseStockModel):
    """CNN Seq2seq模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 kernel_size: int = 3,
                 **kwargs):
        """
        初始化CNN Seq2seq模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: CNN层数
            dropout: dropout概率
            kernel_size: 卷积核大小
        """
        self.kernel_size = kernel_size
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        kernel_size=kernel_size, **kwargs)
        
    def _build_model(self):
        """构建CNN Seq2seq模型架构"""
        # 编码器CNN层
        self.encoder_convs = nn.ModuleList()
        in_channels = self.input_size
        
        for i in range(self.num_layers):
            out_channels = self.hidden_size if i == 0 else self.hidden_size
            self.encoder_convs.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            ))
            in_channels = out_channels
        
        # 解码器CNN层
        self.decoder_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.decoder_convs.append(nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            ))
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 未使用（保持接口一致性）
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        # 转换为CNN格式 (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # 编码器
        for conv in self.encoder_convs:
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 解码器
        for conv in self.decoder_convs:
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch_size, hidden_size)
        
        # 输出层
        output = self.output_layer(x)
        
        return output


class DilatedCNNModel(BaseStockModel):
    """扩张CNN模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 kernel_size: int = 3,
                 **kwargs):
        """
        初始化扩张CNN模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: CNN层数
            dropout: dropout概率
            kernel_size: 卷积核大小
        """
        self.kernel_size = kernel_size
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        kernel_size=kernel_size, **kwargs)
        
    def _build_model(self):
        """构建扩张CNN模型架构"""
        # 输入投影层
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # 扩张卷积层
        self.dilated_convs = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2 ** i  # 扩张率递增
            self.dilated_convs.append(nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) * dilation // 2,
                dilation=dilation
            ))
        
        # 残差连接的层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 未使用（保持接口一致性）
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # 转换为CNN格式
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        
        # 扩张卷积层
        for i, (conv, norm) in enumerate(zip(self.dilated_convs, self.layer_norms)):
            residual = x
            x = conv(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # 残差连接
            if x.shape == residual.shape:
                x = x + residual
            
            # 层归一化（需要转换维度）
            x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
            x = norm(x)
            x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch_size, hidden_size)
        
        # 输出层
        output = self.output_layer(x)
        
        return output


class CNNLSTMModel(BaseStockModel):
    """CNN-LSTM混合模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 kernel_size: int = 3,
                 lstm_layers: int = 2,
                 **kwargs):
        """
        初始化CNN-LSTM混合模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: CNN层数
            dropout: dropout概率
            kernel_size: 卷积核大小
            lstm_layers: LSTM层数
        """
        self.kernel_size = kernel_size
        self.lstm_layers = lstm_layers
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        kernel_size=kernel_size, lstm_layers=lstm_layers, **kwargs)
        
    def _build_model(self):
        """构建CNN-LSTM混合模型架构"""
        # CNN特征提取层
        self.cnn_layers = nn.ModuleList()
        in_channels = self.input_size
        
        for i in range(self.num_layers):
            out_channels = self.hidden_size // 2 if i == 0 else self.hidden_size // 2
            self.cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            ))
            in_channels = out_channels
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.hidden_size // 2,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: LSTM隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        # CNN特征提取
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        
        for cnn in self.cnn_layers:
            x = cnn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # 转换回LSTM格式
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size//2)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        return output


class TemporalCNNModel(BaseStockModel):
    """时间卷积网络模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 6,
                 dropout: float = 0.2,
                 kernel_size: int = 3,
                 **kwargs):
        """
        初始化时间卷积网络模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: TCN层数
            dropout: dropout概率
            kernel_size: 卷积核大小
        """
        self.kernel_size = kernel_size
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        kernel_size=kernel_size, **kwargs)
        
    def _build_model(self):
        """构建时间卷积网络模型架构"""
        # 输入投影层
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # TCN层
        self.tcn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(self._make_tcn_block(
                self.hidden_size, self.hidden_size, self.kernel_size, dilation
            ))
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
    def _make_tcn_block(self, in_channels: int, out_channels: int, 
                       kernel_size: int, dilation: int) -> nn.Module:
        """创建TCN块"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=(kernel_size - 1) * dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, 
                     padding=(kernel_size - 1) * dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，shape: (batch_size, seq_len, input_size)
            hidden: 未使用（保持接口一致性）
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 转换为CNN格式
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        
        # TCN层
        for tcn_block in self.tcn_layers:
            residual = x
            x = tcn_block(x)
            
            # 残差连接
            if x.shape == residual.shape:
                x = x + residual
        
        # 取最后一个时间步的输出
        x = x[:, :, -1]  # (batch_size, hidden_size)
        
        # 输出层
        output = self.output_layer(x)
        
        return output 