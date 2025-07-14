"""
股价预测器

用于模型推理和未来价格预测的预测器类
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from ..data.preprocessor import StockPreprocessor
from ..training.metrics import StockMetrics


class StockPredictor:
    """股价预测器"""
    
    def __init__(self, 
                 model: nn.Module,
                 preprocessor: Optional[StockPreprocessor] = None,
                 device: str = 'auto',
                 logger: Optional[logging.Logger] = None):
        """
        初始化预测器
        
        Args:
            model: 训练好的模型
            preprocessor: 数据预处理器
            device: 设备类型
            logger: 日志记录器
        """
        self.model = model
        self.preprocessor = preprocessor
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 日志记录
        self.logger = logger or self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('StockPredictor')
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def predict(self, 
                data: Union[np.ndarray, torch.Tensor],
                return_prob: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        单步预测
        
        Args:
            data: 输入数据
            return_prob: 是否返回概率
            
        Returns:
            预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).to(self.device)
            else:
                data = data.to(self.device)
            
            # 确保数据维度正确
            if len(data.shape) == 2:
                data = data.unsqueeze(0)  # 添加batch维度
            
            # 预测
            output = self.model(data)
            
            # 转换为numpy数组
            predictions = output.cpu().numpy()
            
            if return_prob:
                # 计算置信度（这里使用简单的方法）
                confidence = np.ones_like(predictions) * 0.8  # 简化的置信度
                return predictions, confidence
            else:
                return predictions
    
    def predict_sequence(self, 
                        data: Union[np.ndarray, torch.Tensor],
                        future_steps: int,
                        method: str = 'recursive') -> np.ndarray:
        """
        多步预测
        
        Args:
            data: 输入数据
            future_steps: 预测步数
            method: 预测方法 ('recursive', 'direct')
            
        Returns:
            预测序列
        """
        self.model.eval()
        
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        predictions = []
        
        if method == 'recursive':
            # 递归预测
            current_input = data.clone()
            
            for step in range(future_steps):
                with torch.no_grad():
                    pred = self.model(current_input)
                    predictions.append(pred.cpu().numpy())
                    
                    # 更新输入序列（滑动窗口）
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        pred.unsqueeze(1)
                    ], dim=1)
        
        elif method == 'direct':
            # 直接预测（需要模型支持）
            with torch.no_grad():
                if hasattr(self.model, 'predict_sequence'):
                    predictions = self.model.predict_sequence(data, future_steps, self.device)
                else:
                    self.logger.warning("模型不支持直接多步预测，回退到递归预测")
                    return self.predict_sequence(data, future_steps, 'recursive')
        
        return np.array(predictions)
    
    def predict_with_confidence(self, 
                              data: Union[np.ndarray, torch.Tensor],
                              n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        带置信区间的预测（蒙特卡洛dropout）
        
        Args:
            data: 输入数据
            n_samples: 采样次数
            
        Returns:
            (预测均值, 预测标准差)
        """
        self.model.train()  # 启用dropout
        
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(data)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        self.model.eval()  # 恢复评估模式
        
        return mean_pred, std_pred
    
    def predict_from_file(self, 
                         data_path: str,
                         output_path: Optional[str] = None,
                         future_steps: int = 1) -> Dict:
        """
        从文件预测
        
        Args:
            data_path: 数据文件路径
            output_path: 输出文件路径
            future_steps: 预测步数
            
        Returns:
            预测结果字典
        """
        if self.preprocessor is None:
            raise ValueError("需要预处理器来处理文件数据")
        
        self.logger.info(f"从文件预测: {data_path}")
        
        # 处理数据
        data_dict = self.preprocessor.process_file(data_path)
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # 预测
        predictions = []
        for i in range(len(X_test)):
            pred = self.predict(X_test[i:i+1])
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # 反归一化
        if self.preprocessor.normalize:
            predictions = self.preprocessor.inverse_transform(predictions)
            y_test = self.preprocessor.inverse_transform(y_test)
        
        # 计算指标
        metrics = StockMetrics.calculate_all_metrics(y_test[:, 0], predictions[:, 0])
        
        # 预测未来
        if future_steps > 1:
            last_sequence = X_test[-1:] 
            future_predictions = self.predict_sequence(last_sequence, future_steps)
            
            if self.preprocessor.normalize:
                future_predictions = self.preprocessor.inverse_transform(future_predictions.reshape(-1, 1))
                future_predictions = future_predictions.flatten()
        else:
            future_predictions = None
        
        # 构建结果
        result = {
            'predictions': predictions,
            'actuals': y_test,
            'metrics': metrics,
            'future_predictions': future_predictions,
            'data_info': {
                'data_path': data_path,
                'test_samples': len(X_test),
                'future_steps': future_steps,
                'prediction_time': datetime.now().isoformat()
            }
        }
        
        # 保存结果
        if output_path:
            self.save_predictions(result, output_path)
        
        self.logger.info(f"预测完成，准确率: {metrics['accuracy']:.2f}%")
        
        return result
    
    def save_predictions(self, predictions: Dict, output_path: str):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典
            output_path: 输出文件路径
        """
        # 创建DataFrame
        df_results = pd.DataFrame({
            'actual': predictions['actuals'][:, 0],
            'predicted': predictions['predictions'][:, 0],
            'error': predictions['actuals'][:, 0] - predictions['predictions'][:, 0]
        })
        
        # 保存预测结果
        df_results.to_csv(output_path, index=False)
        
        # 保存指标
        metrics_path = output_path.replace('.csv', '_metrics.json')
        import json
        # 转换numpy数据类型为Python原生类型
        json_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in predictions['metrics'].items()}
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        self.logger.info(f"预测结果已保存至: {output_path}")
        self.logger.info(f"指标已保存至: {metrics_path}")
    
    def batch_predict(self, 
                     data_paths: List[str],
                     output_dir: str = 'predictions',
                     future_steps: int = 1) -> Dict:
        """
        批量预测
        
        Args:
            data_paths: 数据文件路径列表
            output_dir: 输出目录
            future_steps: 预测步数
            
        Returns:
            批量预测结果
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = {}
        
        for i, data_path in enumerate(data_paths):
            try:
                self.logger.info(f"批量预测 {i+1}/{len(data_paths)}: {data_path}")
                
                # 生成输出文件名
                filename = os.path.basename(data_path).replace('.csv', '_predictions.csv')
                output_path = os.path.join(output_dir, filename)
                
                # 预测
                result = self.predict_from_file(data_path, output_path, future_steps)
                
                batch_results[data_path] = result
                
            except Exception as e:
                self.logger.error(f"预测失败 {data_path}: {str(e)}")
                batch_results[data_path] = {'error': str(e)}
        
        # 保存批量预测摘要
        summary = self._create_batch_summary(batch_results)
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        
        import json
        # 转换numpy数据类型为Python原生类型
        json_summary = {k: float(v) if hasattr(v, 'item') else v for k, v in summary.items()}
        with open(summary_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        self.logger.info(f"批量预测完成，摘要已保存至: {summary_path}")
        
        return batch_results
    
    def _create_batch_summary(self, batch_results: Dict) -> Dict:
        """创建批量预测摘要"""
        successful_predictions = 0
        failed_predictions = 0
        total_accuracy = 0
        
        for data_path, result in batch_results.items():
            if 'error' in result:
                failed_predictions += 1
            else:
                successful_predictions += 1
                total_accuracy += result['metrics']['accuracy']
        
        avg_accuracy = total_accuracy / successful_predictions if successful_predictions > 0 else 0
        
        summary = {
            'total_files': len(batch_results),
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'average_accuracy': avg_accuracy,
            'prediction_time': datetime.now().isoformat()
        }
        
        return summary
    
    def simulate_trading(self, 
                        data_path: str,
                        initial_capital: float = 10000,
                        transaction_cost: float = 0.001) -> Dict:
        """
        模拟交易
        
        Args:
            data_path: 数据文件路径
            initial_capital: 初始资金
            transaction_cost: 交易成本
            
        Returns:
            交易结果
        """
        # 获取预测结果
        result = self.predict_from_file(data_path)
        predictions = result['predictions']
        actuals = result['actuals']
        
        # 模拟交易
        capital = initial_capital
        position = 0  # 持仓数量
        trades = []
        
        for i in range(len(predictions) - 1):
            current_price = actuals[i, 0]
            predicted_price = predictions[i, 0]
            next_actual_price = actuals[i + 1, 0]
            
            # 交易决策
            if predicted_price > current_price and position == 0:
                # 买入
                shares = capital / current_price
                position = shares
                capital = 0
                
                trade_cost = shares * current_price * transaction_cost
                capital -= trade_cost
                
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'cost': trade_cost,
                    'index': i
                })
                
            elif predicted_price < current_price and position > 0:
                # 卖出
                capital = position * current_price
                
                trade_cost = capital * transaction_cost
                capital -= trade_cost
                
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': position,
                    'cost': trade_cost,
                    'index': i
                })
                
                position = 0
        
        # 计算最终资产
        final_price = actuals[-1, 0]
        final_capital = capital + position * final_price
        
        # 计算交易指标
        total_return = (final_capital - initial_capital) / initial_capital * 100
        num_trades = len(trades)
        
        trading_result = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'trades': trades,
            'buy_and_hold_return': (final_price - actuals[0, 0]) / actuals[0, 0] * 100
        }
        
        return trading_result 