"""
Seq2seq系列模型

基于PyTorch实现的序列到序列股价预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_model import BaseStockModel


class Seq2SeqModel(BaseStockModel):
    """基础Seq2seq模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 rnn_type: str = 'LSTM',
                 **kwargs):
        """
        初始化Seq2seq模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: RNN层数
            dropout: dropout概率
            rnn_type: RNN类型 ('LSTM', 'GRU')
        """
        self.rnn_type = rnn_type
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        rnn_type=rnn_type, **kwargs)
        
    def _build_model(self):
        """构建Seq2seq模型架构"""
        # 编码器
        if self.rnn_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        elif self.rnn_type == 'GRU':
            self.encoder = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # 解码器
        if self.rnn_type == 'LSTM':
            self.decoder = nn.LSTM(
                input_size=self.output_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        elif self.rnn_type == 'GRU':
            self.decoder = nn.GRU(
                input_size=self.output_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
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
            hidden: 隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 编码器
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        # 解码器输入（使用编码器的最后输出）
        decoder_input = encoder_outputs[:, -1:, :self.output_size]  # (batch_size, 1, output_size)
        
        # 解码器
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)
        
        # 输出层
        output = self.output_layer(decoder_output.squeeze(1))  # (batch_size, output_size)
        
        return output


class Seq2SeqVAEModel(BaseStockModel):
    """Seq2seq VAE模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 latent_size: int = 64,
                 rnn_type: str = 'LSTM',
                 **kwargs):
        """
        初始化Seq2seq VAE模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: RNN层数
            dropout: dropout概率
            latent_size: 潜在空间维度
            rnn_type: RNN类型 ('LSTM', 'GRU')
        """
        self.rnn_type = rnn_type
        self.latent_size = latent_size
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        rnn_type=rnn_type, latent_size=latent_size, **kwargs)
        
    def _build_model(self):
        """构建Seq2seq VAE模型架构"""
        # 编码器
        if self.rnn_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        elif self.rnn_type == 'GRU':
            self.encoder = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # VAE参数层
        self.mu_layer = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_size, self.latent_size)
        
        # 潜在空间到隐藏状态的映射
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size)
        
        # 解码器
        if self.rnn_type == 'LSTM':
            self.decoder = nn.LSTM(
                input_size=self.latent_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        elif self.rnn_type == 'GRU':
            self.decoder = nn.GRU(
                input_size=self.latent_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
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
        
        # 编码器
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        # 取最后一个时间步的输出
        last_hidden = encoder_outputs[:, -1, :]  # (batch_size, hidden_size)
        
        # VAE参数
        mu = self.mu_layer(last_hidden)
        logvar = self.logvar_layer(last_hidden)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)  # (batch_size, latent_size)
        
        # 解码器输入
        decoder_input = z.unsqueeze(1)  # (batch_size, 1, latent_size)
        
        # 解码器隐藏状态
        decoder_hidden = self.latent_to_hidden(z)  # (batch_size, hidden_size)
        
        if self.rnn_type == 'LSTM':
            # LSTM需要(h0, c0)
            decoder_hidden = (
                decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1),
                decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            )
        else:
            # GRU只需要h0
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 解码器
        decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
        
        # 输出层
        output = self.output_layer(decoder_output.squeeze(1))  # (batch_size, output_size)
        
        return output, mu, logvar
        
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE损失函数"""
        # 重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss


class AttentionSeq2SeqModel(BaseStockModel):
    """带注意力机制的Seq2seq模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 attention_size: int = 64,
                 **kwargs):
        """
        初始化带注意力的Seq2seq模型
        
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
        """构建带注意力的Seq2seq模型架构"""
        # 编码器
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=self.hidden_size,  # 使用注意力输出作为输入
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
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
            hidden: 隐藏状态
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # 编码器
        encoder_outputs, encoder_hidden = self.encoder(x)  # (batch_size, seq_len, hidden_size)
        
        # 注意力权重
        attention_weights = self.attention(encoder_outputs)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(encoder_outputs * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        # 解码器输入
        decoder_input = context.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # 解码器
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)
        
        # 输出层
        output = self.output_layer(decoder_output.squeeze(1))  # (batch_size, output_size)
        
        return output 