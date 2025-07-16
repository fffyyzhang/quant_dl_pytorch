"""
模型训练模块

包含训练器、损失函数、优化器配置等
"""

from .trainer import StockTrainer
from .losses import StockLoss
from .metrics import StockMetrics

__all__ = ['StockTrainer', 'StockLoss', 'StockMetrics'] 