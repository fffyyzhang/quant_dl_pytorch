#!/usr/bin/env python3
"""
简洁的ETF数据处理使用示例

展示如何使用新的简洁ETF数据处理模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from stock_prediction_pytorch.data.etf_simple_processor import ETFProcessor
from torch.utils.data import DataLoader
import pickle


def main():
    """主函数演示完整流程"""
    
    print("=== ETF数据处理示例 ===")
    
    # 1. 处理ETF数据
    processor = ETFProcessor(wnd_size=30)  # 使用30天的历史数据
    
    # 假设你有ETF数据CSV文件
    csv_path = "path/to/your/etf_data.csv"  # 请替换为实际的CSV文件路径
    
    # 如果文件不存在，创建一个示例
    if not os.path.exists(csv_path):
        print("创建示例ETF数据...")
        create_sample_etf_data(csv_path)
    
    # 处理数据
    results = processor.process_etf_data(
        csv_path=csv_path,
        train_ratio=0.8,  # 80%训练，20%测试
        output_dir="processed_etf_data",
        min_samples=100  # 至少需要100个交易日的数据
    )
    
    if not results:
        print("没有成功处理任何ETF数据")
        return
    
    # 2. 加载处理好的数据
    first_etf = list(results.keys())[0]
    print(f"\n使用 {first_etf} 的数据进行演示")
    
    # 直接加载数据集
    with open(results[first_etf]['train_path'], 'rb') as f:
        train_dataset = pickle.load(f)
    with open(results[first_etf]['test_path'], 'rb') as f:
        test_dataset = pickle.load(f)
    
    print(f"训练数据: {train_dataset.X.shape} -> {train_dataset.y.shape}")
    print(f"测试数据: {test_dataset.X.shape} -> {test_dataset.y.shape}")
    
    # 3. 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 简单的LSTM模型示例
    class SimpleETFModel(nn.Module):
        def __init__(self, input_size=5, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # 取最后一个时间步的输出
            output = self.fc(lstm_out[:, -1, :])
            return output.squeeze()
    
    model = SimpleETFModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 简单训练循环
    print("\n开始训练...")
    model.train()
    for epoch in range(5):  # 只训练5个epoch作为演示
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, 平均损失: {avg_loss:.6f}")
    
    # 6. 测试预测
    print("\n开始测试...")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # 7. 显示结果 (归一化后的值)
    print(f"\n预测结果 (前10个, 归一化值):")
    for i in range(min(10, len(predictions))):
        print(f"实际: {actuals[i]:.4f}, 预测: {predictions[i]:.4f}")
    print("\n注意: 显示的是归一化后的值，反归一化应在模型预测阶段处理")
    
    print("\n完成! 这就是使用新ETF数据处理模块的完整流程。")


def create_sample_etf_data(csv_path: str):
    """创建示例ETF数据"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建示例数据
    dates = []
    data = []
    
    start_date = datetime(2023, 1, 1)
    base_price = 100.0
    
    for i in range(300):  # 300个交易日
        date = start_date + timedelta(days=i)
        
        # 模拟价格变动
        change = np.random.normal(0, 0.02)  # 2%的标准差
        base_price *= (1 + change)
        
        # 生成OHLC数据
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = np.random.uniform(10000, 100000)
        
        data.append({
            'ts_code': '159001.SZ',
            'trade_date': date.strftime('%Y%m%d'),
            'pre_close': base_price / (1 + change),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'change': close_price - (base_price / (1 + change)),
            'pct_chg': change * 100,
            'vol': volume,
            'amount': volume * close_price,
            'stock_name': '货币ETF'
        })
    
    # 创建第二个ETF的数据
    base_price = 50.0
    for i in range(300):
        date = start_date + timedelta(days=i)
        
        change = np.random.normal(0, 0.025)  # 稍微高一点的波动
        base_price *= (1 + change)
        
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = np.random.uniform(5000, 50000)
        
        data.append({
            'ts_code': '510300.SH',
            'trade_date': date.strftime('%Y%m%d'),
            'pre_close': base_price / (1 + change),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'change': close_price - (base_price / (1 + change)),
            'pct_chg': change * 100,
            'vol': volume,
            'amount': volume * close_price,
            'stock_name': '沪深300ETF'
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"示例数据已保存到: {csv_path}")


if __name__ == "__main__":
    main() 