"""
工具函数模块

包含各种辅助函数和工具
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd


def set_seed(seed: int = 42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    获取可用设备
    
    Returns:
        设备对象
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_info(model: torch.nn.Module, filepath: str):
    """
    保存模型信息
    
    Args:
        model: PyTorch模型
        filepath: 保存路径
    """
    info = {
        'model_name': model.__class__.__name__,
        'total_params': count_parameters(model),
        'model_structure': str(model),
        'save_time': datetime.now().isoformat()
    }
    
    import json
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)


def create_directory(path: str):
    """
    创建目录
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    获取文件大小
    
    Args:
        filepath: 文件路径
        
    Returns:
        文件大小字符串
    """
    if not os.path.exists(filepath):
        return "文件不存在"
    
    size = os.path.getsize(filepath)
    
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.1f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.1f} MB"
    else:
        return f"{size/1024**3:.1f} GB"


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        return f"{seconds//60:.0f}分{seconds%60:.0f}秒"
    else:
        return f"{seconds//3600:.0f}小时{(seconds%3600)//60:.0f}分{seconds%60:.0f}秒"


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        show: 是否显示图片
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
    if 'val_loss' in history and any(loss > 0 for loss in history['val_loss']):
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('轮数')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='训练准确率', color='blue')
    if 'val_acc' in history:
        axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('轮数')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 损失对比
    if 'val_loss' in history and any(loss > 0 for loss in history['val_loss']):
        axes[1, 0].plot(history['train_loss'], label='训练损失', alpha=0.7)
        axes[1, 0].plot(history['val_loss'], label='验证损失', alpha=0.7)
        axes[1, 0].set_title('训练 vs 验证损失')
        axes[1, 0].set_xlabel('轮数')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 准确率对比
    if 'val_acc' in history:
        axes[1, 1].plot(history['train_acc'], label='训练准确率', alpha=0.7)
        axes[1, 1].plot(history['val_acc'], label='验证准确率', alpha=0.7)
        axes[1, 1].set_title('训练 vs 验证准确率')
        axes[1, 1].set_xlabel('轮数')
        axes[1, 1].set_ylabel('准确率 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(actual: np.ndarray, 
                    predicted: np.ndarray,
                    title: str = "股价预测结果",
                    save_path: Optional[str] = None,
                    show: bool = True):
    """
    绘制预测结果
    
    Args:
        actual: 实际值
        predicted: 预测值
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图片
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(actual, label='实际价格', color='blue', linewidth=2)
    plt.plot(predicted, label='预测价格', color='red', linewidth=2, linestyle='--')
    
    plt.title(title, fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    plt.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_matrix(data: pd.DataFrame, 
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    绘制相关性矩阵
    
    Args:
        data: 数据框
        save_path: 保存路径
        show: 是否显示图片
    """
    plt.figure(figsize=(10, 8))
    
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    
    plt.title('特征相关性矩阵', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性矩阵图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_price_distribution(prices: np.ndarray,
                          title: str = "价格分布",
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    绘制价格分布
    
    Args:
        prices: 价格数组
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图片
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 价格时间序列
    axes[0].plot(prices, color='blue', linewidth=1.5)
    axes[0].set_title('价格时间序列')
    axes[0].set_xlabel('时间')
    axes[0].set_ylabel('价格')
    axes[0].grid(True, alpha=0.3)
    
    # 价格分布直方图
    axes[1].hist(prices, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1].set_title('价格分布')
    axes[1].set_xlabel('价格')
    axes[1].set_ylabel('频次')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"价格分布图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def calculate_technical_indicators(prices: pd.Series) -> pd.DataFrame:
    """
    计算技术指标
    
    Args:
        prices: 价格序列
        
    Returns:
        技术指标DataFrame
    """
    df = pd.DataFrame()
    df['Price'] = prices
    
    # 移动平均线
    df['MA_5'] = prices.rolling(window=5).mean()
    df['MA_10'] = prices.rolling(window=10).mean()
    df['MA_20'] = prices.rolling(window=20).mean()
    
    # 指数移动平均线
    df['EMA_5'] = prices.ewm(span=5).mean()
    df['EMA_10'] = prices.ewm(span=10).mean()
    df['EMA_20'] = prices.ewm(span=20).mean()
    
    # 布林带
    df['BB_Upper'] = df['MA_20'] + 2 * prices.rolling(window=20).std()
    df['BB_Lower'] = df['MA_20'] - 2 * prices.rolling(window=20).std()
    
    # RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 价格变化率
    df['Price_Change'] = prices.pct_change()
    df['Price_Change_Abs'] = prices.diff()
    
    return df


def validate_data(data: np.ndarray) -> bool:
    """
    验证数据有效性
    
    Args:
        data: 数据数组
        
    Returns:
        是否有效
    """
    if data is None or len(data) == 0:
        return False
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False
    
    return True


def normalize_data(data: np.ndarray, 
                  method: str = 'minmax') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    数据归一化
    
    Args:
        data: 输入数据
        method: 归一化方法 ('minmax', 'zscore')
        
    Returns:
        (归一化数据, 归一化参数)
    """
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val}
    elif method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized = (data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val}
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized, params


def denormalize_data(normalized_data: np.ndarray, 
                    params: Dict[str, float],
                    method: str = 'minmax') -> np.ndarray:
    """
    反归一化
    
    Args:
        normalized_data: 归一化数据
        params: 归一化参数
        method: 归一化方法
        
    Returns:
        原始数据
    """
    if method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    elif method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def print_system_info():
    """打印系统信息"""
    print("=== 系统信息 ===")
    import sys
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("===============") 