"""
ETF数据集构建示例

演示如何使用ETF数据集构建功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append('/home/liyuan/proj_liy/quant_dl_pytorch')

from stock_prediction_pytorch.data import (
    ETFDatasetFactory, 
    build_etf_dataset,
    ETFDataReader
)


def main():
    """主函数"""
    # ETF数据文件路径
    data_file = '/data/data_liy/quant/raw/etf_daily.csv'
    
    print("=== ETF数据集构建示例 ===\n")
    
    # 1. 查看数据基本信息
    print("1. 数据基本信息:")
    factory = ETFDatasetFactory(data_file)
    info = factory.get_data_info()
    print(f"总记录数: {info['total_records']}")
    print(f"ETF数量: {info['total_symbols']}")
    print(f"时间范围: {info['date_range']['start']} 到 {info['date_range']['end']}")
    print(f"样本ETF: {info['symbols'][:5]}")
    print()
    
    # 2. 构建单个ETF数据集
    print("2. 构建单个ETF数据集:")
    ts_code = '159001.SZ'  # 货币ETF
    
    try:
        train_dataset, test_dataset = factory.build_single_etf_dataset(
            ts_code=ts_code,
            window_size=20,
            train_ratio=0.9
        )
        
        print(f"ETF {ts_code}:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        print(f"  序列长度: {train_dataset.get_sequence_length()}")
        print(f"  特征维度: {train_dataset.get_feature_dim()}")
        print(f"  特征名称: {train_dataset.feature_names[:5]}...")
        print()
        
    except Exception as e:
        print(f"构建单ETF数据集失败: {e}")
        print()
    
    # 3. 构建多ETF数据集
    print("3. 构建多ETF数据集:")
    try:
        # 选择前10个ETF进行测试
        reader = ETFDataReader(data_file)
        available_codes = reader.get_available_symbols()
        test_codes = available_codes[:10]
        
        multi_dataset = factory.build_multi_etf_dataset(
            ts_codes=test_codes,
            window_size=20,
            train_ratio=0.9,
            min_samples=200  # 需要至少200个样本
        )
        
        dataset_info = multi_dataset.get_info()
        print(f"成功处理的ETF数量: {dataset_info['total_etfs']}")
        print(f"总训练样本: {dataset_info['total_train_samples']}")
        print(f"总测试样本: {dataset_info['total_test_samples']}")
        print(f"特征维度: {dataset_info['features']}")
        print()
        
    except Exception as e:
        print(f"构建多ETF数据集失败: {e}")
        print()
    
    # 4. 使用便捷函数一键构建
    print("4. 一键构建数据加载器:")
    try:
        train_loader, test_loader = build_etf_dataset(
            data_file_path=data_file,
            ts_codes=['159001.SZ'],  # 单个ETF
            window_size=15,
            train_ratio=0.8,
            batch_size=16
        )
        
        print(f"训练数据加载器: {len(train_loader)} 批次")
        print(f"测试数据加载器: {len(test_loader)} 批次")
        
        # 查看一个批次的数据
        for batch_X, batch_y in train_loader:
            print(f"批次数据形状: X{batch_X.shape}, y{batch_y.shape}")
            break
        print()
        
    except Exception as e:
        print(f"一键构建失败: {e}")
        print()
    
    # 5. 构建平衡数据集
    print("5. 构建平衡数据集:")
    try:
        # 使用前5个ETF构建平衡数据集
        balanced_dataset = factory.build_balanced_dataset(
            ts_codes=available_codes[:5],
            window_size=20,
            train_ratio=0.9,
            max_samples_per_etf=100,  # 每个ETF最多100个样本
            min_samples=150
        )
        
        balanced_info = balanced_dataset.get_info()
        print(f"平衡数据集ETF数量: {balanced_info['total_etfs']}")
        print(f"平衡后训练样本: {balanced_info['total_train_samples']}")
        print(f"平衡后测试样本: {balanced_info['total_test_samples']}")
        
        # 显示每个ETF的样本数
        for ts_code, stats in balanced_info['per_etf_stats'].items():
            print(f"  {ts_code}: 训练{stats['train_samples']}, 测试{stats['test_samples']}")
        
    except Exception as e:
        print(f"构建平衡数据集失败: {e}")
    
    print("\n=== 示例完成 ===")


if __name__ == '__main__':
    main() 