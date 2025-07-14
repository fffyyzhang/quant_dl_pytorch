# 使用示例

本目录包含股价预测PyTorch模型包的各种使用示例。

## 示例文件

### 📚 基础示例

- **`basic_usage.py`** - 基础使用示例
  - 展示数据加载、模型训练、预测的完整流程
  - 适合初学者了解包的基本功能

### 🎯 运行示例

#### 基础使用示例

```bash
cd examples/
python basic_usage.py
```

**注意**: 示例需要参考数据文件，请确保 `ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv` 文件存在。

#### 命令行示例

```bash
# 训练模型
python ../main.py --mode train --data ../ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv --output ./output/

# 预测价格
python ../main.py --mode predict --data ../ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv --model ./output/best_model.pth --output ./predictions/

# 运行完整示例
python ../main.py --mode example --data ../ref_codes/Stock-Prediction-Models/dataset/GOOG-year.csv --model-type LSTM --output ./example_output/
```

## 🔧 自定义示例

您可以根据需要创建自己的示例：

1. 复制 `basic_usage.py`
2. 修改数据路径、模型参数等
3. 运行您的自定义示例

## 📊 示例数据

示例使用Google股票数据 (`GOOG-year.csv`)，包含：
- Date: 日期
- Open: 开盘价
- High: 最高价
- Low: 最低价
- Close: 收盘价
- Volume: 成交量

## 🎨 可视化示例

运行示例后，您可以使用可视化工具查看结果：

```python
from stock_prediction_pytorch.utils.helpers import plot_predictions, plot_training_history
import pandas as pd

# 读取预测结果
df = pd.read_csv('basic_predictions.csv')
plot_predictions(df['actual'], df['predicted'], save_path='prediction_plot.png')
```

## 📝 注意事项

1. 确保已安装所有依赖：`pip install -r ../requirements.txt`
2. 确保数据文件路径正确
3. 第一次运行可能需要较长时间进行模型训练
4. 示例使用较少的训练轮数，实际使用时可增加轮数以获得更好效果

## 🚀 下一步

- 尝试不同的模型类型（GRU, Transformer等）
- 调整超参数优化模型性能
- 使用自己的股票数据进行训练
- 实现自定义的损失函数和评估指标 