"""
评估指标模块

股价预测模型的评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StockMetrics:
    """股价预测评估指标"""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算准确率（参考原始代码）
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            准确率百分比
        """
        y_true = np.array(y_true) + 1
        y_pred = np.array(y_pred) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
        return percentage * 100
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对百分比误差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MAPE值
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算均方根误差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            RMSE值
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对误差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MAE值
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算R²分数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            R²值
        """
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算方向准确率
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            方向准确率
        """
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        correct_directions = np.sum(true_direction == pred_direction)
        total_predictions = len(y_true)
        
        return (correct_directions / total_predictions) * 100
    
    @staticmethod
    def calculate_trend_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算趋势准确率
        
        Args:
            y_true: 真实值序列
            y_pred: 预测值序列
            
        Returns:
            趋势准确率
        """
        if len(y_true) < 2:
            return 0.0
        
        true_trends = np.diff(y_true)
        pred_trends = np.diff(y_pred)
        
        true_trend_directions = np.sign(true_trends)
        pred_trend_directions = np.sign(pred_trends)
        
        correct_trends = np.sum(true_trend_directions == pred_trend_directions)
        total_trends = len(true_trends)
        
        return (correct_trends / total_trends) * 100
    
    @staticmethod
    def calculate_volatility_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算波动率准确率
        
        Args:
            y_true: 真实值序列
            y_pred: 预测值序列
            
        Returns:
            波动率准确率
        """
        true_vol = np.std(y_true)
        pred_vol = np.std(y_pred)
        
        vol_error = abs(true_vol - pred_vol) / true_vol
        vol_accuracy = (1 - vol_error) * 100
        
        return max(0, vol_accuracy)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = StockMetrics.calculate_accuracy(y_true, y_pred)
        metrics['rmse'] = StockMetrics.calculate_rmse(y_true, y_pred)
        metrics['mae'] = StockMetrics.calculate_mae(y_true, y_pred)
        metrics['mape'] = StockMetrics.calculate_mape(y_true, y_pred)
        metrics['r2'] = StockMetrics.calculate_r2(y_true, y_pred)
        
        # 方向性指标
        metrics['directional_accuracy'] = StockMetrics.calculate_directional_accuracy(y_true, y_pred)
        metrics['trend_accuracy'] = StockMetrics.calculate_trend_accuracy(y_true, y_pred)
        metrics['volatility_accuracy'] = StockMetrics.calculate_volatility_accuracy(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """
        打印评估指标
        
        Args:
            metrics: 指标字典
        """
        print("=== 模型评估指标 ===")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.2f}%")
        print(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
        print(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {metrics['mape']:.2f}%")
        print(f"R²分数: {metrics['r2']:.4f}")
        print(f"方向准确率: {metrics['directional_accuracy']:.2f}%")
        print(f"趋势准确率: {metrics['trend_accuracy']:.2f}%")
        print(f"波动率准确率: {metrics['volatility_accuracy']:.2f}%")
        print("==================")


class TradingMetrics:
    """交易评估指标"""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """
        计算收益率
        
        Args:
            prices: 价格序列
            
        Returns:
            收益率序列
        """
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险收益率
            
        Returns:
            夏普比率
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            最大回撤
        """
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)
    
    @staticmethod
    def calculate_win_rate(predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        计算胜率
        
        Args:
            predictions: 预测价格变化方向
            actual: 实际价格变化方向
            
        Returns:
            胜率
        """
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        
        correct_predictions = np.sum(pred_direction == actual_direction)
        total_predictions = len(predictions)
        
        return (correct_predictions / total_predictions) * 100
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """
        计算盈利因子
        
        Args:
            returns: 收益率序列
            
        Returns:
            盈利因子
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        total_profit = np.sum(positive_returns)
        total_loss = abs(np.sum(negative_returns))
        
        return total_profit / total_loss if total_loss != 0 else float('inf')
    
    @staticmethod
    def calculate_trading_metrics(prices: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        计算交易指标
        
        Args:
            prices: 实际价格序列
            predictions: 预测价格序列
            
        Returns:
            交易指标字典
        """
        actual_returns = TradingMetrics.calculate_returns(prices)
        pred_returns = TradingMetrics.calculate_returns(predictions)
        
        metrics = {
            'sharpe_ratio': TradingMetrics.calculate_sharpe_ratio(actual_returns),
            'max_drawdown': TradingMetrics.calculate_max_drawdown(prices),
            'win_rate': TradingMetrics.calculate_win_rate(pred_returns, actual_returns),
            'profit_factor': TradingMetrics.calculate_profit_factor(actual_returns),
            'total_return': (prices[-1] - prices[0]) / prices[0] * 100,
            'volatility': np.std(actual_returns) * np.sqrt(252) * 100  # 年化波动率
        }
        
        return metrics 