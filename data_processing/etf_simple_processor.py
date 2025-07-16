"""
简洁的ETF数据处理模块

处理CSV格式的ETF数据，生成PyTorch可直接使用的数据集
避免过度封装，以最简方式实现功能
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import os
from typing import Tuple, Dict, List, Optional


class ETFProcessor:
    """简洁的ETF数据处理器"""
    
    def __init__(self, wnd_size: int = 30, filter_list: List[str] = None):
        """
        Args:
            wnd_size: 历史时间窗口大小
        """
        self.wnd_size = wnd_size
        self.filter_list = filter_list
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        #过滤掉filter_list中的etf
        if self.filter_list:
            df = df[df['ts_code'].isin(self.filter_list)]
        
        # 确保有必要的列
        required_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 转换日期格式 (YYYYMMDD -> datetime)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 删除包含NaN的行
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'vol'])
        
        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'vol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除转换失败的行
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'vol'])
        
        return df
    
    def _process_single_etf(self, 
                           etf_data: pd.DataFrame, 
                           ts_code: str, 
                           train_ratio: float) -> Dict:
        """处理单个ETF的数据"""
        # 提取OHLCV数据
        ohlcv_data = etf_data[['open', 'high', 'low', 'close', 'vol']].values
        
        # 创建时间序列数据 - 在每个窗口内归一化
        X, y, timestamps = [], [], []
        
        for i in range(len(ohlcv_data) - self.wnd_size):
            # 当前窗口数据
            window_data = ohlcv_data[i:i+self.wnd_size]
            next_close = ohlcv_data[i+self.wnd_size, 3]  # 下一个时间点的close价格
            
            # 获取y对应的时间戳（被预测值的时间）
            y_timestamp = etf_data.iloc[i+self.wnd_size]['trade_date']
            if hasattr(y_timestamp, 'strftime'):
                y_timestamp_str = y_timestamp.strftime('%Y-%m-%d')
            else:
                y_timestamp_str = str(y_timestamp)
            
            # 在窗口内归一化
            scaler = MinMaxScaler()
            window_normalized = scaler.fit_transform(window_data)
            
            # 对目标值也进行归一化 (使用同一个scaler的close价格部分)
            # 创建一个dummy数组来归一化目标值
            dummy_target = np.zeros((1, 5))
            dummy_target[0, 3] = next_close  # close价格在第3列
            dummy_normalized = scaler.transform(dummy_target)
            target_normalized = dummy_normalized[0, 3]
            
            X.append(window_normalized)
            y.append(target_normalized)
            timestamps.append(y_timestamp_str)
        
        X, y = np.array(X), np.array(y)
        
        # 分割训练测试集
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        timestamps_train, timestamps_test = timestamps[:split_idx], timestamps[split_idx:]
        
        # 返回字典，包含原始numpy数组和时间戳
        return {
            'X_train': X_train,
            'y_train': y_train,
            'timestamps_train': timestamps_train,
            'X_test': X_test,
            'y_test': y_test,
            'timestamps_test': timestamps_test
        }
    
    
    def process_etf_data(self, 
                        csv_path: str, 
                        train_ratio: float = 0.8,
                        output_dir: str = 'processed_data',
                        min_samples: int = 100) -> Dict[str, str]:
        """
        处理ETF数据的主函数
        
        Args:
            csv_path: CSV文件路径
            train_ratio: 训练集比例
            output_dir: 输出目录
            min_samples: 每个ETF最少需要的样本数
            
        Returns:
            Dict: 包含处理结果信息的字典
        """
        print(f"开始处理ETF数据: {csv_path}")
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        print(f"原始数据行数: {len(df)}")
        
        # 数据预处理
        df = self._preprocess_data(df)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 按ETF分组处理
        results = {}
        etf_codes = df['ts_code'].unique()
        print(f"发现 {len(etf_codes)} 个ETF代码")
        
        processed_count = 0
        for ts_code in etf_codes:
            etf_data = df[df['ts_code'] == ts_code].sort_values('trade_date')
            
            if len(etf_data) < min_samples:
                print(f"跳过 {ts_code}: 数据不足 ({len(etf_data)} < {min_samples})")
                continue
                
            try:
                data = self._process_single_etf(
                    etf_data, ts_code, train_ratio
                )
                
                # 保存数据集
                train_path = os.path.join(output_dir, f'{ts_code}_train.pkl')
                test_path = os.path.join(output_dir, f'{ts_code}_test.pkl')
                
                # 保存训练数据
                train_data = {
                    'X': data['X_train'], 
                    'y': data['y_train'],
                    'timestamps': data['timestamps_train']
                }
                with open(train_path, 'wb') as f:
                    pickle.dump(train_data, f)
                
                # 保存测试数据
                test_data = {
                    'X': data['X_test'], 
                    'y': data['y_test'],
                    'timestamps': data['timestamps_test']
                }
                with open(test_path, 'wb') as f:
                    pickle.dump(test_data, f)
                
                results[ts_code] = {
                    'train_path': train_path,
                    'test_path': test_path,
                    'train_samples': len(data['X_train']),
                    'test_samples': len(data['X_test'])
                }
                
                processed_count += 1
                print(f"已处理 {ts_code}: 训练集{len(data['X_train'])}样本, 测试集{len(data['X_test'])}样本")
                
            except Exception as e:
                print(f"处理 {ts_code} 失败: {e}")
                continue
        
        # 保存处理结果信息为JSON格式
        info_path = os.path.join(output_dir, 'process_info.json')
        with open(info_path, 'w', encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成! 成功处理 {processed_count} 个ETF")
        print(f"数据保存在: {output_dir}")
        return results


def test_processed_data(results_or_json_path):
    """测试已处理的ETF数据
    
    Args:
        results_or_json_path: 可以是处理结果字典，或者是process_info.json文件路径
    """
    with open(results_or_json_path, 'r', encoding='utf8') as f:
        results = json.load(f)

    print(f"找到 {len(results)} 个已处理的ETF")
    
    # 选择第一个ETF进行测试
    first_etf = list(results.keys())[0]
    print(f"使用 {first_etf} 进行测试")
    
    try:
        # 直接加载数据
        with open(results[first_etf]['train_path'], 'rb') as f:
            train_data = pickle.load(f)
        with open(results[first_etf]['test_path'], 'rb') as f:
            test_data = pickle.load(f)
        
        # 检查新的数据格式（包含时间戳）
        train_timestamps = train_data['timestamps']
        test_timestamps = test_data['timestamps']
        
        # 创建torch tensors和DataLoader
        train_X = torch.FloatTensor(train_data['X'])
        train_y = torch.FloatTensor(train_data['y'])
        test_X = torch.FloatTensor(test_data['X'])
        test_y = torch.FloatTensor(test_data['y'])
        
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 测试数据加载
        train_batch_X, train_batch_y = next(iter(train_loader))
        test_batch_X, test_batch_y = next(iter(test_loader))
        
        print(f"✓ 训练数据形状: {train_X.shape} -> {train_y.shape}")
        print(f"✓ 测试数据形状: {test_X.shape} -> {test_y.shape}")
        print(f"✓ 训练时间戳数量: {len(train_timestamps)}")
        print(f"✓ 测试时间戳数量: {len(test_timestamps)}")
        print(f"✓ 训练批次形状: {train_batch_X.shape} -> {train_batch_y.shape}")
        print(f"✓ 测试批次形状: {test_batch_X.shape} -> {test_batch_y.shape}")
        
        # 显示前几个时间戳示例
        print(f"✓ 训练时间戳示例: {train_timestamps[:3]}")
        print(f"✓ 测试时间戳示例: {test_timestamps[:3]}")
        
        print("✓ 数据加载测试通过，可以开始训练模型!")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        return False


if __name__ == "__main__":
    
    #处理数据
    
    df=pd.read_csv("/data/data_liy/quant/etf/etf_human_select.csv",encoding='utf8')
    human_select_list=df[df.human==1]['ts_code'].tolist()
    
    processor = ETFProcessor(wnd_size=30, filter_list=human_select_list)
    results = processor.process_etf_data(
        csv_path="/data/data_liy/quant/raw/etf_daily.csv",
        train_ratio=0.8,
        output_dir="/data/data_liy/quant/train/etf/etf_wnd_30",
        min_samples=50
    )
    
    #测试dataloader
    test_processed_data("/data/data_liy/quant/train/etf/etf_wnd_30/process_info.json") 