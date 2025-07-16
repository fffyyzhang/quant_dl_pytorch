"""
自定义损失函数

股价预测专用的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockLoss(nn.Module):
    """股价预测专用损失函数"""
    
    def __init__(self, loss_type: str = 'mse', alpha: float = 0.5):
        """
        初始化损失函数
        
        Args:
            loss_type: 损失类型 ('mse', 'mae', 'huber', 'directional')
            alpha: 组合损失的权重
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 预测值
            target: 真实值
            
        Returns:
            损失值
        """
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'huber':
            return F.huber_loss(pred, target)
        elif self.loss_type == 'directional':
            return self._directional_loss(pred, target)
        elif self.loss_type == 'combined':
            return self._combined_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _directional_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """方向性损失（惩罚预测方向错误）"""
        # 计算价格变化方向
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)
        
        # 方向一致性损失
        directional_loss = 1.0 - torch.mean(pred_direction * target_direction)
        
        # 结合MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        return mse_loss + self.alpha * directional_loss
    
    def _combined_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """组合损失（MSE + MAE）"""
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss


class TrendLoss(nn.Module):
    """趋势损失函数"""
    
    def __init__(self, trend_weight: float = 0.3):
        """
        初始化趋势损失
        
        Args:
            trend_weight: 趋势损失权重
        """
        super().__init__()
        self.trend_weight = trend_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算趋势损失
        
        Args:
            pred: 预测值
            target: 真实值
            
        Returns:
            损失值
        """
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 趋势损失
        if pred.size(1) > 1:  # 多步预测
            pred_trend = pred[:, 1:] - pred[:, :-1]
            target_trend = target[:, 1:] - target[:, :-1]
            trend_loss = F.mse_loss(pred_trend, target_trend)
        else:  # 单步预测
            trend_loss = torch.tensor(0.0, device=pred.device)
        
        return mse_loss + self.trend_weight * trend_loss


class QuantileLoss(nn.Module):
    """分位数损失函数"""
    
    def __init__(self, quantile: float = 0.5):
        """
        初始化分位数损失
        
        Args:
            quantile: 分位数（0-1之间）
        """
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算分位数损失
        
        Args:
            pred: 预测值
            target: 真实值
            
        Returns:
            损失值
        """
        errors = target - pred
        loss = torch.maximum(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return torch.mean(loss)


class VolatilityLoss(nn.Module):
    """波动率损失函数"""
    
    def __init__(self, vol_weight: float = 0.2):
        """
        初始化波动率损失
        
        Args:
            vol_weight: 波动率损失权重
        """
        super().__init__()
        self.vol_weight = vol_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算波动率损失
        
        Args:
            pred: 预测值
            target: 真实值
            
        Returns:
            损失值
        """
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 波动率损失
        pred_vol = torch.std(pred, dim=1)
        target_vol = torch.std(target, dim=1)
        vol_loss = F.mse_loss(pred_vol, target_vol)
        
        return mse_loss + self.vol_weight * vol_loss 