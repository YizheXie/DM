# 金融欺诈检测系统

基于机器学习和深度学习方法的金融欺诈检测综合系统，包含三个主要任务：多类欺诈分类、欺诈异常检测和欺诈相关元数据时间序列预测。

## 功能模块

### 1. 多类欺诈检测

使用两种分类方法执行多类欺诈检测：
- XGBoost分类器
- 随机森林分类器

这个模块将欺诈检测与设备类型相结合，创建多类标签（如手机欺诈、ATM欺诈等）。

### 2. 欺诈异常检测

使用无监督学习方法进行欺诈异常检测：
- 基于深度学习的自编码器
- K-means聚类（聚为两类：高风险和低风险）

### 3. 时间序列预测

预测与欺诈检测相关的元数据信息：
- LSTM深度学习模型
- 随机森林回归/分类

## 安装与使用

### 环境要求

```bash
pip install -r requirements.txt
```

### 使用方法

1. 运行完整系统：
```bash
python main.py --data_path path/to/dataset.csv --sample_size 100000
```

2. 仅运行特定任务：
```bash
python main.py --task classification  # 仅运行多类欺诈检测
python main.py --task anomaly         # 仅运行异常检测
python main.py --task time_series     # 仅运行时间序列预测
```

3. 超参数优化：
```bash
python main.py --task classification --optimize
```

4. 选择时间序列预测目标：
```bash
python main.py --task time_series --target_feature device_used
```

## 数据集

使用的金融交易数据集包含以下主要特征：
- 交易ID、时间戳、金额等基本信息
- 发送方和接收方账户
- 交易类型和商家类别
- 地理位置和设备信息
- 欺诈标签和风险评分

## 实现细节

1. **类别不平衡处理**：使用SMOTE过采样技术处理欺诈样本少的问题。

2. **多类别标签创建**：结合欺诈状态与设备类型/支付渠道创建多类别标签。

3. **评估指标**：使用准确率、精确率、召回率、F1分数、ROC-AUC等多种指标评估模型性能。

4. **可视化**：提供混淆矩阵、ROC曲线、重要特征、聚类结果等多种可视化。