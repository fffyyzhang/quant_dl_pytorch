"""
Models Package - 股价预测模型包

提供各种深度学习模型的统一接口和创建功能
"""

# 导入所有模型类
from .lstm_models import LSTMModel, BiLSTMModel, MultiLSTMModel, LSTM2PathModel, LSTMAttentionModel
from .gru_models import GRUModel, BiGRUModel, MultiGRUModel  
from .cnn_models import CNNSeq2SeqModel, DilatedCNNModel, CNNLSTMModel, TemporalCNNModel
from .attention_models import TransformerModel, AttentionModel
from .seq2seq_models import Seq2SeqModel, Seq2SeqVAEModel, AttentionSeq2SeqModel
from .base_model import BaseStockModel, StockRNNBase, StockLSTMBase, StockTransformerBase


# 模型映射字典 - 简单的字典映射方式
MODEL_MAP = {
    'lstm': LSTMModel,
    'bilstm': BiLSTMModel,
    'multilstm': MultiLSTMModel,
    'lstm2path': LSTM2PathModel,
    'lstmattention': LSTMAttentionModel,
    
    'gru': GRUModel,
    'bigru': BiGRUModel,
    'multigru': MultiGRUModel,
    
    'cnn': CNNSeq2SeqModel,
    'dilated_cnn': DilatedCNNModel,
    'cnnlstm': CNNLSTMModel,
    'temporal_cnn': TemporalCNNModel,
    
    'transformer': TransformerModel,
    'attention': AttentionModel,
    
    'seq2seq': Seq2SeqModel,
    'seq2seqvae': Seq2SeqVAEModel,
    'attention_seq2seq': AttentionSeq2SeqModel,
}


def create_model(model_name: str, **kwargs):
    """
    根据模型名称创建模型实例
    
    Args:
        model_name: 模型名称 (lstm, gru, cnn, transformer等)
        **kwargs: 模型参数
        
    Returns:
        模型实例
        
    Example:
        >>> from models import create_model
        >>> model = create_model('lstm', input_size=5, hidden_size=64, output_size=1)
        >>> model = create_model('transformer', input_size=5, hidden_size=128, num_heads=8)
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_MAP:
        available = list(MODEL_MAP.keys())
        raise ValueError(f"不支持的模型: {model_name}，可用模型: {available}")
    
    model_class = MODEL_MAP[model_name]
    return model_class(**kwargs)


def get_available_models():
    """获取所有可用的模型名称"""
    return list(MODEL_MAP.keys())


def get_model_info():
    """获取模型信息"""
    info = {}
    for name, model_class in MODEL_MAP.items():
        info[name] = {
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'base_class': model_class.__bases__[0].__name__ if model_class.__bases__ else 'object'
        }
    return info


def get_models_by_category():
    """按类别获取模型列表"""
    categories = {
        'LSTM': [name for name in MODEL_MAP.keys() if 'lstm' in name],
        'GRU': [name for name in MODEL_MAP.keys() if 'gru' in name],
        'CNN': [name for name in MODEL_MAP.keys() if 'cnn' in name],
        'Transformer': [name for name in MODEL_MAP.keys() if any(x in name for x in ['transformer', 'attention'])],
        'Seq2Seq': [name for name in MODEL_MAP.keys() if 'seq2seq' in name],
    }
    return categories


# 包的公共接口
__all__ = [
    # 模型创建函数
    'create_model',
    'get_available_models', 
    'get_model_info',
    'get_models_by_category',
    
    # 基础模型类
    'BaseStockModel',
    'StockRNNBase', 
    'StockLSTMBase',
    'StockTransformerBase',
    
    # LSTM系列
    'LSTMModel',
    'BiLSTMModel', 
    'MultiLSTMModel',
    'LSTM2PathModel',
    'LSTMAttentionModel',
    
    # GRU系列
    'GRUModel',
    'BiGRUModel',
    'MultiGRUModel',
    
    # CNN系列
    'CNNSeq2SeqModel',
    'DilatedCNNModel',
    'CNNLSTMModel', 
    'TemporalCNNModel',
    
    # Transformer系列
    'TransformerModel',
    'AttentionModel',
    
    # Seq2Seq系列
    'Seq2SeqModel',
    'Seq2SeqVAEModel',
    'AttentionSeq2SeqModel',
] 