"""
股价数据集类

PyTorch Dataset实现，用于时间序列数据加载
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class StockDataset(Dataset):
    """股价时间序列数据集"""
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            X: 输入序列数据，shape: (n_samples, sequence_length, n_features)
            y: 目标数据，shape: (n_samples, n_targets)
            transform: 数据变换函数
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.X.shape[-1]
    
    def get_sequence_length(self) -> int:
        """获取序列长度"""
        return self.X.shape[1]
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        return self.y.shape[-1] if len(self.y.shape) > 1 else 1


class StockDatasetFromFile(Dataset):
    """从文件直接加载的股价数据集"""
    
    def __init__(self, 
                 data_path: str,
                 target_column: str = 'Close',
                 feature_columns: Optional[list] = None,
                 sequence_length: int = 5,
                 test_size: int = 30,
                 train: bool = True,
                 normalize: bool = True):
        """
        从文件初始化数据集
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            feature_columns: 特征列名列表
            sequence_length: 序列长度
            test_size: 测试集大小
            train: 是否为训练集
            normalize: 是否归一化
        """
        from .preprocessor import StockPreprocessor
        
        self.preprocessor = StockPreprocessor(
            target_column=target_column,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            test_size=test_size,
            normalize=normalize
        )
        
        # 处理数据
        data_dict = self.preprocessor.process_file(data_path)
        
        if train:
            self.X = torch.FloatTensor(data_dict['X_train'])
            self.y = torch.FloatTensor(data_dict['y_train'])
        else:
            self.X = torch.FloatTensor(data_dict['X_test'])
            self.y = torch.FloatTensor(data_dict['y_test'])
            
        self.original_df = data_dict['original_df']
        self.normalized_data = data_dict['normalized_data']
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_preprocessor(self):
        """获取预处理器"""
        return self.preprocessor
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.X.shape[-1]
    
    def get_sequence_length(self) -> int:
        """获取序列长度"""
        return self.X.shape[1]
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        return self.y.shape[-1] if len(self.y.shape) > 1 else 1


class MultiStockDataset(Dataset):
    """多股票数据集"""
    
    def __init__(self, datasets: list):
        """
        初始化多股票数据集
        
        Args:
            datasets: StockDataset对象列表
        """
        self.datasets = datasets
        self.cumulative_lengths = []
        
        cumsum = 0
        for dataset in datasets:
            cumsum += len(dataset)
            self.cumulative_lengths.append(cumsum)
            
        self.total_length = cumsum
        
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 找到对应的数据集
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                break
                
        # 计算在该数据集中的索引
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
            
        return self.datasets[dataset_idx][local_idx]
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.datasets[0].get_feature_dim()
    
    def get_sequence_length(self) -> int:
        """获取序列长度"""
        return self.datasets[0].get_sequence_length()
    
    def get_target_dim(self) -> int:
        """获取目标维度"""
        return self.datasets[0].get_target_dim() 