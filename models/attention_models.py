"""
Transformer和Attention模型

基于PyTorch实现的Transformer股价预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .base_model import StockTransformerBase


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑和输出投影
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerModel(StockTransformerBase):
    """Transformer模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 **kwargs):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: dropout概率
            max_len: 最大序列长度
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        num_heads, max_len=max_len, **kwargs)
        
    def _build_model(self):
        """构建Transformer模型架构"""
        # 输入投影层
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.hidden_size, self.model_config['max_len'])
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.hidden_size,
                num_heads=self.num_heads,
                d_ff=self.hidden_size * 4,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)
        
        # Dropout层
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
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, hidden_size)
        
        # Dropout
        x = self.dropout_layer(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output


class AttentionModel(StockTransformerBase):
    """简化的注意力模型（参考原始代码）"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs):
        """
        初始化注意力模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_layers: 注意力层数
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout, 
                        num_heads, **kwargs)
        
    def _build_model(self):
        """构建注意力模型架构"""
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.input_size, self.hidden_size)
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.hidden_size, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
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
            hidden: 未使用（保持接口一致性）
            
        Returns:
            输出预测，shape: (batch_size, output_size)
        """
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 多层注意力
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # 自注意力
            attn_output = attention(x, x, x)
            x = norm(x + self.dropout_layer(attn_output))
            
            # 前馈网络
            if i == len(self.attention_layers) - 1:  # 最后一层
                ff_output = self.feed_forward(x)
                x = norm(x + self.dropout_layer(ff_output))
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 输出层
        output = self.output_layer(x)
        
        return output


def create_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建注意力掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0 