"""
股价数据预处理器

基于参考代码实现的数据预处理功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class StockPreprocessor:
    """股价数据预处理器"""
    
    def __init__(self, 
                 target_column: str = 'Close',
                 feature_columns: Optional[List[str]] = None,
                 sequence_length: int = 5,
                 test_size: int = 30,
                 normalize: bool = True):
        """
        初始化预处理器
        
        Args:
            target_column: 目标列名（默认为收盘价）
            feature_columns: 特征列名列表，如果为None则只使用target_column
            sequence_length: 时间序列长度
            test_size: 测试集大小（天数）
            normalize: 是否进行归一化
        """
        self.target_column = target_column
        self.feature_columns = feature_columns or [target_column]
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.normalize = normalize
        
        # 缩放器
        self.scaler = MinMaxScaler() if normalize else None
        self.is_fitted = False
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载股价数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            DataFrame: 加载的数据
        """
        df = pd.read_csv(data_path)
        
        # 检查必要的列是否存在
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
            
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        准备数据，包括归一化和格式转换
        
        Args:
            df: 原始数据框
            
        Returns:
            tuple: (处理后的数据, 原始数据框)
        """
        # 提取特征列
        feature_data = df[self.feature_columns].astype('float32')
        
        if self.normalize:
            # 归一化
            if not self.is_fitted:
                normalized_data = self.scaler.fit_transform(feature_data)
                self.is_fitted = True
            else:
                normalized_data = self.scaler.transform(feature_data)
        else:
            normalized_data = feature_data.values
            
        return normalized_data, df
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            data: 输入数据
            
        Returns:
            tuple: (X, y) 序列数据和标签
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            # 输入序列
            X.append(data[i:(i + self.sequence_length)])
            # 目标值（下一个时间步）
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def split_train_test(self, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分割训练集和测试集
        
        Args:
            X: 输入序列
            y: 目标值
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = len(X) - self.test_size
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化
        
        Args:
            data: 归一化的数据
            
        Returns:
            反归一化后的数据
        """
        if self.normalize and self.is_fitted:
            if len(data.shape) == 1:
                # 单列数据
                data_reshaped = data.reshape(-1, 1)
                result = self.scaler.inverse_transform(data_reshaped)
                return result.flatten()
            else:
                # 多列数据
                return self.scaler.inverse_transform(data)
        else:
            return data
    
    def process_file(self, data_path: str) -> dict:
        """
        处理单个数据文件的完整流程
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            dict: 包含处理后数据的字典
        """
        # 加载数据
        df = self.load_data(data_path)
        
        # 准备数据
        normalized_data, original_df = self.prepare_data(df)
        
        # 创建序列
        X, y = self.create_sequences(normalized_data)
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'original_df': original_df,
            'normalized_data': normalized_data,
            'scaler': self.scaler
        }
    
    def calculate_accuracy(self, real: np.ndarray, predict: np.ndarray) -> float:
        """
        计算预测准确率（参考原始代码）
        
        Args:
            real: 真实值
            predict: 预测值
            
        Returns:
            准确率百分比
        """
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100
    
    def anchor_smoothing(self, signal: np.ndarray, weight: float = 0.3) -> np.ndarray:
        """
        锚点平滑（参考原始代码）
        
        Args:
            signal: 输入信号
            weight: 平滑权重
            
        Returns:
            平滑后的信号
        """
        buffer = []
        last = signal[0]
        
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
            
        return np.array(buffer) 