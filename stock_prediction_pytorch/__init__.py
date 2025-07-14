"""
PyTorch股价预测模型包

基于深度学习的股价预测模型，包含多种架构：
- LSTM系列模型
- GRU系列模型  
- Transformer/Attention模型
- CNN系列模型
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction PyTorch Team"

# 导入主要模块
from .models import *
from .data import *
from .training import *
from .prediction import *
from .utils import *

__all__ = [
    'models',
    'data', 
    'training',
    'prediction',
    'utils'
] 