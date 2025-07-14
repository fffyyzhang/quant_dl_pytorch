"""
数据加载器模块

创建和配置PyTorch DataLoader的工厂函数
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from .dataset import StockDataset, StockDatasetFromFile, MultiStockDataset


def create_data_loader(dataset: torch.utils.data.Dataset,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      pin_memory: bool = True,
                      drop_last: bool = False) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否锁定内存
        drop_last: 是否丢弃最后不完整的批次
        
    Returns:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_train_test_loaders(X_train: torch.Tensor,
                             y_train: torch.Tensor,
                             X_test: torch.Tensor,
                             y_test: torch.Tensor,
                             batch_size: int = 32,
                             num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        X_train: 训练集输入
        y_train: 训练集标签
        X_test: 测试集输入
        y_test: 测试集标签
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """
    # 创建数据集
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def create_loaders_from_file(data_path: str,
                            target_column: str = 'Close',
                            feature_columns: Optional[list] = None,
                            sequence_length: int = 5,
                            test_size: int = 30,
                            batch_size: int = 32,
                            normalize: bool = True,
                            num_workers: int = 0) -> Tuple[DataLoader, DataLoader, object]:
    """
    从文件创建训练和测试数据加载器
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名
        feature_columns: 特征列名列表
        sequence_length: 序列长度
        test_size: 测试集大小
        batch_size: 批次大小
        normalize: 是否归一化
        num_workers: 工作进程数
        
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """
    # 创建数据集
    train_dataset = StockDatasetFromFile(
        data_path=data_path,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        test_size=test_size,
        train=True,
        normalize=normalize
    )
    
    test_dataset = StockDatasetFromFile(
        data_path=data_path,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        test_size=test_size,
        train=False,
        normalize=normalize
    )
    
    # 创建数据加载器
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, train_dataset.get_preprocessor()


def create_multi_stock_loader(data_paths: list,
                             target_column: str = 'Close',
                             feature_columns: Optional[list] = None,
                             sequence_length: int = 5,
                             test_size: int = 30,
                             batch_size: int = 32,
                             train: bool = True,
                             normalize: bool = True,
                             num_workers: int = 0) -> DataLoader:
    """
    创建多股票数据加载器
    
    Args:
        data_paths: 数据文件路径列表
        target_column: 目标列名
        feature_columns: 特征列名列表
        sequence_length: 序列长度
        test_size: 测试集大小
        batch_size: 批次大小
        train: 是否为训练集
        normalize: 是否归一化
        num_workers: 工作进程数
        
    Returns:
        DataLoader: 多股票数据加载器
    """
    datasets = []
    
    for data_path in data_paths:
        dataset = StockDatasetFromFile(
            data_path=data_path,
            target_column=target_column,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            test_size=test_size,
            train=train,
            normalize=normalize
        )
        datasets.append(dataset)
    
    # 创建多股票数据集
    multi_dataset = MultiStockDataset(datasets)
    
    # 创建数据加载器
    loader = create_data_loader(
        multi_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    
    return loader 