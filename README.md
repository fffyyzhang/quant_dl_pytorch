# 股价预测PyTorch模型包

基于PyTorch实现的股价预测深度学习模型包，提供多种先进的神经网络架构用于时间序列预测。

## 🚀 特性

- **多种模型架构**：LSTM、GRU、Transformer、CNN等18种深度学习模型
- **完整的训练流程**：数据预处理、模型训练、验证、保存一体化
- **灵活的配置系统**：支持JSON/YAML配置文件，便于管理超参数
- **强大的预测功能**：支持单步/多步预测、置信区间估计、批量预测
- **丰富的评估指标**：准确率、RMSE、MAE、方向准确率等多种指标
- **可视化工具**：训练曲线、预测结果、相关性分析等图表
- **易于使用**：提供命令行接口和Python API两种使用方式

## 📦 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/stock-prediction-pytorch.git
cd stock-prediction-pytorch
pip install -e .
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 支持的模型

### LSTM系列
- **LSTMModel**: 基础LSTM模型
- **BiLSTMModel**: 双向LSTM模型
- **MultiLSTMModel**: 多层LSTM模型
- **LSTM2PathModel**: 双路径LSTM模型
- **LSTMAttentionModel**: 带注意力机制的LSTM

### GRU系列
- **GRUModel**: 基础GRU模型
- **BiGRUModel**: 双向GRU模型
- **MultiGRUModel**: 多层GRU模型

### Transformer系列
- **TransformerModel**: 完整Transformer模型
- **AttentionModel**: 简化注意力模型

### CNN系列
- **CNNSeq2SeqModel**: CNN序列到序列模型
- **DilatedCNNModel**: 扩张CNN模型
- **CNNLSTMModel**: CNN-LSTM混合模型
- **TemporalCNNModel**: 时间卷积网络

### Seq2seq系列
- **Seq2SeqModel**: 基础序列到序列模型
- **Seq2SeqVAEModel**: 变分自编码器Seq2seq模型
- **AttentionSeq2SeqModel**: 带注意力的Seq2seq模型

## 🛠️ 使用方法

### 1. 命令行使用

#### 训练模型

```bash
# 使用默认LSTM模型训练
python main.py --mode train --data data/GOOG.csv --output checkpoints/

# 使用Transformer模型训练
python main.py --mode train --data data/GOOG.csv --model-type Transformer --output checkpoints/

# 使用配置文件训练
python main.py --mode train --data data/GOOG.csv --config config.json --output checkpoints/
```

#### 预测价格

```bash
# 使用训练好的模型预测
python main.py --mode predict --data data/GOOG.csv --model checkpoints/best_model.pth --output predictions/

# 预测未来30天价格
python main.py --mode predict --data data/GOOG.csv --model checkpoints/best_model.pth --future-steps 30
```

#### 运行完整示例

```bash
# 运行LSTM完整示例
python main.py --mode example --data data/GOOG.csv --model-type LSTM --output example_output/
```

### 2. Python API使用

#### 基础使用

```python
from stock_prediction_pytorch import *

# 创建配置
config = Config()
config.model.model_type = 'LSTM'
config.model.hidden_size = 128
config.training.num_epochs = 100
config.data.sequence_length = 10

# 加载数据
train_loader, val_loader, preprocessor = create_loaders_from_file(
    data_path='data/GOOG.csv',
    batch_size=32,
    sequence_length=10
)

# 创建模型
model = LSTMModel(
    input_size=1,
    hidden_size=128,
    output_size=1,
    num_layers=2
)

# 训练模型
trainer = StockTrainer(model, train_loader, val_loader)
history = trainer.train(num_epochs=100)

# 预测
predictor = StockPredictor(model, preprocessor)
result = predictor.predict_from_file('data/GOOG.csv')
```

#### 高级使用

```python
# 使用自定义损失函数
from stock_prediction_pytorch.training.losses import StockLoss

loss_fn = StockLoss(loss_type='directional', alpha=0.3)

# 使用多种评估指标
from stock_prediction_pytorch.training.metrics import StockMetrics

metrics = StockMetrics.calculate_all_metrics(y_true, y_pred)
StockMetrics.print_metrics(metrics)

# 带置信区间的预测
mean_pred, std_pred = predictor.predict_with_confidence(data, n_samples=100)

# 模拟交易
trading_result = predictor.simulate_trading('data/GOOG.csv', initial_capital=10000)
```

## 📊 数据格式

支持标准的股价CSV文件格式：

```csv
Date,Open,High,Low,Close,Volume
2021-01-01,100.0,102.0,99.0,101.0,1000000
2021-01-02,101.0,103.0,100.0,102.0,1100000
...
```

- **必需列**: Date, Close
- **可选列**: Open, High, Low, Volume
- **日期格式**: 支持多种标准日期格式

## ⚙️ 配置文件

### JSON配置示例

```json
{
  "model": {
    "model_type": "LSTM",
    "input_size": 1,
    "hidden_size": 128,
    "output_size": 1,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": false
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "optimizer": "Adam",
    "loss_function": "MSE",
    "patience": 10
  },
  "data": {
    "target_column": "Close",
    "sequence_length": 10,
    "test_size": 30,
    "normalize": true
  }
}
```

### YAML配置示例

```yaml
model:
  model_type: "Transformer"
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1

training:
  learning_rate: 0.0001
  batch_size: 16
  num_epochs: 200
  optimizer: "Adam"
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 200

data:
  sequence_length: 20
  test_size: 60
  normalize: true
```

## 📈 评估指标

模型提供多种评估指标：

- **准确率 (Accuracy)**: 基于相对误差的准确率
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **方向准确率**: 价格变化方向预测准确率
- **趋势准确率**: 价格趋势预测准确率
- **波动率准确率**: 价格波动预测准确率

## 🎨 可视化功能

```python
from stock_prediction_pytorch.utils.helpers import *

# 绘制训练历史
plot_training_history(history, save_path='training_history.png')

# 绘制预测结果
plot_predictions(actual, predicted, save_path='predictions.png')

# 绘制相关性矩阵
plot_correlation_matrix(data, save_path='correlation.png')

# 绘制价格分布
plot_price_distribution(prices, save_path='distribution.png')
```

## 🔧 自定义模型

您可以轻松创建自定义模型：

```python
from stock_prediction_pytorch.models.base_model import BaseStockModel
import torch.nn as nn

class CustomModel(BaseStockModel):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)
    
    def _build_model(self):
        # 定义您的模型架构
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, x, hidden=None):
        # 实现前向传播
        return self.layers(x[:, -1, :])  # 使用最后一个时间步
```

## 📚 示例和教程

查看 `examples/` 目录获取更多使用示例：

- `basic_usage.py`: 基础使用示例
- `advanced_training.py`: 高级训练技巧
- `custom_models.py`: 自定义模型示例
- `batch_prediction.py`: 批量预测示例
- `trading_simulation.py`: 交易模拟示例

## 🚨 注意事项

1. **数据质量**: 确保输入数据质量良好，无缺失值或异常值
2. **参数调优**: 不同股票可能需要不同的超参数设置
3. **过拟合**: 注意监控验证损失，避免过拟合
4. **风险提示**: 本包仅用于研究和学习，不构成投资建议

## 🤝 贡献

欢迎提交问题和功能请求！如果您想贡献代码：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

本项目参考了以下优秀的开源项目：

- [Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) - 提供了丰富的模型实现参考
- PyTorch - 深度学习框架
- scikit-learn - 机器学习工具
- pandas - 数据处理工具

## 📞 联系方式

如有问题，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/yourusername/stock-prediction-pytorch/issues)
- 邮箱: your.email@example.com

---

⭐ 如果这个项目对您有帮助，请给个星星！ 