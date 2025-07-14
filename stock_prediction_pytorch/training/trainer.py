"""
模型训练器

股价预测模型的训练器，包含训练循环、验证、保存等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from datetime import datetime
from tqdm import tqdm
import logging


class StockTrainer:
    """股价预测模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'auto',
                 learning_rate: float = 0.001,
                 optimizer: str = 'Adam',
                 loss_function: str = 'MSE',
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 save_dir: str = 'checkpoints',
                 logger: Optional[logging.Logger] = None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备类型
            learning_rate: 学习率
            optimizer: 优化器类型
            loss_function: 损失函数类型
            patience: 早停的耐心值
            min_delta: 早停的最小改善值
            save_dir: 模型保存目录
            logger: 日志记录器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器设置
        self.optimizer = self._get_optimizer(optimizer, learning_rate)
        
        # 损失函数设置
        self.criterion = self._get_loss_function(loss_function)
        
        # 早停设置
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 保存设置
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch': []
        }
        
        # 日志记录
        self.logger = logger or self._setup_logger()
        
    def _get_optimizer(self, optimizer_name: str, lr: float) -> optim.Optimizer:
        """获取优化器"""
        optimizers = {
            'Adam': optim.Adam(self.model.parameters(), lr=lr),
            'SGD': optim.SGD(self.model.parameters(), lr=lr, momentum=0.9),
            'RMSprop': optim.RMSprop(self.model.parameters(), lr=lr),
            'AdamW': optim.AdamW(self.model.parameters(), lr=lr)
        }
        
        if optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizers[optimizer_name]
    
    def _get_loss_function(self, loss_name: str) -> nn.Module:
        """获取损失函数"""
        losses = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'Huber': nn.HuberLoss(),
            'SmoothL1': nn.SmoothL1Loss()
        }
        
        if loss_name not in losses:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        return losses[loss_name]
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('StockTrainer')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(self.save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 计算准确率
            acc = self._calculate_accuracy(output, target)
            total_acc += acc
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': acc
            })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for data, target in progress_bar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失
                loss = self.criterion(output, target)
                
                # 记录损失
                total_loss += loss.item()
                
                # 计算准确率
                acc = self._calculate_accuracy(output, target)
                total_acc += acc
                
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': acc
                })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """计算准确率（参考原始代码）"""
        output_np = output.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        
        # 对于多维输出，只取第一个维度
        if len(output_np.shape) > 1:
            output_np = output_np[:, 0]
        if len(target_np.shape) > 1:
            target_np = target_np[:, 0]
        
        # 计算准确率
        real = np.array(target_np) + 1
        predict = np.array(output_np) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        
        return percentage * 100
    
    def train(self, 
              num_epochs: int,
              scheduler: Optional[str] = None,
              scheduler_params: Optional[Dict] = None,
              save_best_only: bool = True,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            scheduler: 学习率调度器类型
            scheduler_params: 调度器参数
            save_best_only: 是否只保存最佳模型
            verbose: 是否显示详细信息
            
        Returns:
            训练历史记录
        """
        # 设置学习率调度器
        lr_scheduler = None
        if scheduler:
            lr_scheduler = self._get_scheduler(scheduler, scheduler_params or {})
        
        self.logger.info(f"开始训练，共 {num_epochs} 轮")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"模型: {self.model.__class__.__name__}")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_acc'].append(val_acc)
            
            # 学习率调度
            if lr_scheduler:
                if scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()
            
            # 日志记录
            if verbose:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            
            # 早停检查
            if self._should_early_stop(val_loss if val_loss > 0 else train_loss):
                self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            # 保存模型
            if save_best_only:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(f'best_model_epoch_{epoch+1}.pth')
            else:
                if (epoch + 1) % 10 == 0:  # 每10轮保存一次
                    self.save_model(f'model_epoch_{epoch+1}.pth')
        
        self.logger.info("训练完成")
        return self.train_history
    
    def _get_scheduler(self, scheduler_name: str, params: Dict) -> Any:
        """获取学习率调度器"""
        schedulers = {
            'StepLR': optim.lr_scheduler.StepLR(self.optimizer, **params),
            'MultiStepLR': optim.lr_scheduler.MultiStepLR(self.optimizer, **params),
            'ExponentialLR': optim.lr_scheduler.ExponentialLR(self.optimizer, **params),
            'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params),
            'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        }
        
        if scheduler_name not in schedulers:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return schedulers[scheduler_name]
    
    def _should_early_stop(self, current_loss: float) -> bool:
        """检查是否应该早停"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'best_loss': self.best_loss,
            'model_config': self.model.model_config if hasattr(self.model, 'model_config') else {}
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型已保存至: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def save_training_history(self, filename: str = 'training_history.json'):
        """保存训练历史"""
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        self.logger.info(f"训练历史已保存至: {filepath}")
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.train_history['epoch']:
            return {'message': '尚未开始训练'}
        
        summary = {
            'total_epochs': len(self.train_history['epoch']),
            'best_train_loss': min(self.train_history['train_loss']),
            'best_train_acc': max(self.train_history['train_acc']),
            'final_train_loss': self.train_history['train_loss'][-1],
            'final_train_acc': self.train_history['train_acc'][-1],
        }
        
        if self.train_history['val_loss'] and any(loss > 0 for loss in self.train_history['val_loss']):
            summary.update({
                'best_val_loss': min(loss for loss in self.train_history['val_loss'] if loss > 0),
                'best_val_acc': max(self.train_history['val_acc']),
                'final_val_loss': self.train_history['val_loss'][-1],
                'final_val_acc': self.train_history['val_acc'][-1],
            })
        
        return summary 