import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库
import matplotlib  # 导入绘图基础库
matplotlib.use('Agg')  # 设置matplotlib后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt  # 导入matplotlib的绘图功能
import seaborn as sns  # 导入高级绘图库
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入评估指标
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 导入数据预处理工具
from tensorflow.keras.models import Sequential  # 导入Keras顺序模型
from tensorflow.keras.layers import LSTM, Dense, Dropout  # 导入Keras层
from tensorflow.keras.callbacks import EarlyStopping  # 导入早停回调
import time  # 导入时间库，用于计时
import warnings  # 导入警告控制
import os  # 导入操作系统接口
os.makedirs('plots', exist_ok=True)  # 创建plots文件夹，如果已存在则不报错

warnings.filterwarnings('ignore')  # 忽略所有警告信息

class TimeSeriesPredictor:
    def __init__(self):
        """
        Initialize time series predictor with LSTM and Random Forest models
        """
        self.models = {}  # 初始化存储模型的字典
        self.results = {}  # 初始化存储结果的字典
        self.scalers = {}  # 初始化存储数据缩放器的字典
    
    def prepare_time_series_data(self, X, y, sequence_length=10):
        """
        Transform data into sequences for LSTM model
        """
        # First convert input data to numpy array to avoid Pandas index issues
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):  # 检查X是否为pandas对象
            X_values = X.values  # 转换为numpy数组
        else:
            X_values = np.array(X)  # 如果不是pandas对象，直接转为numpy数组
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):  # 检查y是否为pandas对象
            y_values = y.values  # 转换为numpy数组
        else:
            y_values = np.array(y)  # 如果不是pandas对象，直接转为numpy数组
            
        X_seq, y_seq = [], []  # 初始化序列容器
        for i in range(len(X_values) - sequence_length):  # 遍历数据，创建长度为sequence_length的序列
            X_seq.append(X_values[i:i+sequence_length])  # 添加输入序列
            y_seq.append(y_values[i+sequence_length])  # 添加对应的目标值（序列后的下一个值）
        
        return np.array(X_seq), np.array(y_seq)  # 返回转换为numpy数组的序列数据
        
        # 函数功能总结：将时间序列数据转换为LSTM模型所需的序列格式，每个输入是sequence_length长度的序列，输出是序列后的下一个值。
    
    def train_lstm(self, X_train, y_train, sequence_length=10, units=50, epochs=50, batch_size=32):
        """
        Train LSTM model for time series prediction
        """
        print("Training LSTM for time series prediction...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_time_series_data(X_train, y_train, sequence_length)  # 准备LSTM所需的序列数据
        
        print(f"Prepared {len(X_seq)} sequences of length {sequence_length}")  # 打印序列数据信息
        print(f"Input shape: {X_seq.shape}, Target shape: {y_seq.shape}")  # 打印输入和目标的形状
        
        # Create LSTM model
        model = Sequential([  # 创建顺序模型
            LSTM(units, activation='relu', return_sequences=True,  # 第一个LSTM层，返回序列
                input_shape=(sequence_length, X_train.shape[1])),  # 设置输入形状
            Dropout(0.2),  # 添加Dropout层防止过拟合，丢弃20%的神经元
            LSTM(units//2, activation='relu'),  # 第二个LSTM层，单元数为第一层的一半
            Dropout(0.2),  # 再次添加Dropout层
            Dense(1 if len(y_seq.shape) == 1 else y_seq.shape[1])  # 输出层，根据目标维度决定输出单元数
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')  # 使用Adam优化器和均方误差损失函数编译模型
        
        # Set up early stopping
        early_stopping = EarlyStopping(  # 创建早停回调
            monitor='val_loss',  # 监控验证损失
            patience=5,  # 5个epoch没有改善则停止
            restore_best_weights=True  # 恢复最佳权重
        )
        
        # Train model
        history = model.fit(  # 训练模型
            X_seq, y_seq,  # 输入序列和目标值
            epochs=epochs,  # 训练轮数
            batch_size=batch_size,  # 批次大小
            validation_split=0.2,  # 使用20%的数据作为验证集
            callbacks=[early_stopping],  # 使用早停回调
            verbose=1  # 显示训练进度
        )
        
        # Store model and training history
        self.models['lstm'] = model  # 存储训练好的LSTM模型
        self.results['lstm_history'] = history.history  # 存储训练历史
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"LSTM training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        return model  # 返回训练好的模型
        
        # 函数功能总结：训练LSTM模型用于时间序列预测，包括数据准备、模型构建、编译和训练过程，并存储模型和训练历史记录。
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        """
        Train Random Forest for time series prediction
        """
        print("Training Random Forest for time series prediction...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Create and train model
        model = RandomForestRegressor(  # 创建随机森林回归器
            n_estimators=n_estimators,  # 树的数量
            max_depth=max_depth,  # 树的最大深度，None表示不限制
            random_state=42,  # 设置随机种子保证可重复性
            n_jobs=-1  # 使用所有可用处理器
        )
        
        model.fit(X_train, y_train)  # 训练模型
        
        # Store model
        self.models['random_forest'] = model  # 存储训练好的随机森林模型
        
        # Get feature importance
        importance = dict(zip(  # 创建特征重要性字典
            [f"f{i}" for i in range(X_train.shape[1])],  # 特征名称（f0, f1, ...）
            model.feature_importances_  # 特征重要性值
        ))
        self.results['rf_importance'] = importance  # 存储特征重要性
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"Random Forest training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        return model  # 返回训练好的模型
        
        # 函数功能总结：训练随机森林模型用于时间序列预测，配置模型参数，训练模型，并存储模型和特征重要性信息。
    
    def predict_lstm(self, X_test, y_test, sequence_length=10):
        """
        Make predictions using trained LSTM model
        """
        if 'lstm' not in self.models:  # 检查LSTM模型是否已训练
            raise ValueError("LSTM model not found. Train the model first.")  # 如果未训练则抛出错误
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_time_series_data(X_test, y_test, sequence_length)  # 准备测试数据序列
        
        # Get predictions
        model = self.models['lstm']  # 获取训练好的LSTM模型
        y_pred = model.predict(X_seq)  # 使用模型预测
        
        # Flatten predictions if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:  # 如果预测结果是二维且第二维为1
            y_pred = y_pred.flatten()  # 将预测结果展平为一维
        
        # Store results
        self.results['lstm_true'] = y_seq  # 存储真实值
        self.results['lstm_pred'] = y_pred  # 存储预测值
        
        return y_pred, y_seq  # 返回预测值和真实值
        
        # 函数功能总结：使用训练好的LSTM模型对测试数据进行预测，处理预测结果的形状，并存储预测结果和真实值。
    
    def predict_random_forest(self, X_test, y_test):
        """
        Make predictions using trained Random Forest model
        """
        if 'random_forest' not in self.models:  # 检查随机森林模型是否已训练
            raise ValueError("Random Forest model not found. Train the model first.")  # 如果未训练则抛出错误
        
        # Get predictions
        model = self.models['random_forest']  # 获取训练好的随机森林模型
        y_pred = model.predict(X_test)  # 使用模型预测
        
        # Store results
        self.results['rf_true'] = y_test  # 存储真实值
        self.results['rf_pred'] = y_pred  # 存储预测值
        
        return y_pred  # 返回预测值
        
        # 函数功能总结：使用训练好的随机森林模型对测试数据进行预测，并存储预测结果和真实值。
    
    def evaluate_lstm(self, y_true, y_pred, is_classification=False):
        """
        Evaluate LSTM model performance
        """
        print("Evaluating LSTM model...")  # 打印评估开始的信息
        
        if is_classification:  # 如果是分类任务
            # For classification tasks
            # Round predictions to nearest integer for comparison
            y_pred_classes = np.round(y_pred).astype(int)  # 将预测值四舍五入为整数作为类别
            accuracy = np.mean(y_pred_classes == y_true)  # 计算准确率
            print(f"Accuracy: {accuracy:.4f}")  # 打印准确率
            
            results = {  # 创建评估结果字典
                'accuracy': accuracy,  # 存储准确率
                'mse': mean_squared_error(y_true, y_pred),  # 计算均方误差
                'mae': mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
            }
        else:  # 如果是回归任务
            # For regression tasks
            mse = mean_squared_error(y_true, y_pred)  # 计算均方误差
            rmse = np.sqrt(mse)  # 计算均方根误差
            mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
            r2 = r2_score(y_true, y_pred)  # 计算R平方
            
            print(f"Mean Squared Error: {mse:.4f}")  # 打印均方误差
            print(f"Root Mean Squared Error: {rmse:.4f}")  # 打印均方根误差
            print(f"Mean Absolute Error: {mae:.4f}")  # 打印平均绝对误差
            print(f"R² Score: {r2:.4f}")  # 打印R平方
            
            results = {  # 创建评估结果字典
                'mse': mse,  # 存储均方误差
                'rmse': rmse,  # 存储均方根误差
                'mae': mae,  # 存储平均绝对误差
                'r2': r2  # 存储R平方
            }
        
        self.results['lstm_metrics'] = results  # 存储LSTM模型评估指标
        return results  # 返回评估结果
        
        # 函数功能总结：根据任务类型（分类或回归）评估LSTM模型性能，计算并存储相关评估指标，如准确率、均方误差、均方根误差、平均绝对误差和R平方。
    
    def evaluate_random_forest(self, y_true, y_pred, is_classification=False):
        """
        Evaluate Random Forest model performance
        """
        print("Evaluating Random Forest model...")  # 打印评估开始的信息
        
        if is_classification:  # 如果是分类任务
            # For classification tasks
            # Round predictions to nearest integer for comparison
            y_pred_classes = np.round(y_pred).astype(int)  # 将预测值四舍五入为整数作为类别
            accuracy = np.mean(y_pred_classes == y_true)  # 计算准确率
            print(f"Accuracy: {accuracy:.4f}")  # 打印准确率
            
            results = {  # 创建评估结果字典
                'accuracy': accuracy,  # 存储准确率
                'mse': mean_squared_error(y_true, y_pred),  # 计算均方误差
                'mae': mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
            }
        else:  # 如果是回归任务
            # For regression tasks
            mse = mean_squared_error(y_true, y_pred)  # 计算均方误差
            rmse = np.sqrt(mse)  # 计算均方根误差
            mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
            r2 = r2_score(y_true, y_pred)  # 计算R平方
            
            print(f"Mean Squared Error: {mse:.4f}")  # 打印均方误差
            print(f"Root Mean Squared Error: {rmse:.4f}")  # 打印均方根误差
            print(f"Mean Absolute Error: {mae:.4f}")  # 打印平均绝对误差
            print(f"R² Score: {r2:.4f}")  # 打印R平方
            
            results = {  # 创建评估结果字典
                'mse': mse,  # 存储均方误差
                'rmse': rmse,  # 存储均方根误差
                'mae': mae,  # 存储平均绝对误差
                'r2': r2  # 存储R平方
            }
        
        self.results['rf_metrics'] = results  # 存储随机森林模型评估指标
        return results  # 返回评估结果
        
        # 函数功能总结：根据任务类型（分类或回归）评估随机森林模型性能，计算并存储相关评估指标，如准确率、均方误差、均方根误差、平均绝对误差和R平方。
    
    def plot_predictions(self, model_name, num_samples=100, is_classification=False):
        """
        Plot actual vs predicted values
        """
        if model_name == 'lstm':  # 如果是LSTM模型
            y_true = self.results.get('lstm_true')  # 获取LSTM模型的真实值
            y_pred = self.results.get('lstm_pred')  # 获取LSTM模型的预测值
            title = 'LSTM'  # 设置图表标题
        elif model_name == 'random_forest':  # 如果是随机森林模型
            y_true = self.results.get('rf_true')  # 获取随机森林模型的真实值
            y_pred = self.results.get('rf_pred')  # 获取随机森林模型的预测值
            title = 'Random Forest'  # 设置图表标题
        else:
            raise ValueError(f"Unsupported model: {model_name}")  # 如果模型名称不支持则抛出错误
        
        if y_true is None or y_pred is None:  # 检查是否有预测结果
            raise ValueError(f"No predictions found for {model_name}. Run prediction first.")  # 如果没有预测结果则抛出错误
        
        # Plot a sample of the predictions
        plt.figure(figsize=(12, 6))  # 创建图表，设置大小
        
        # If classification, make a scatter plot
        if is_classification:  # 如果是分类任务
            plt.scatter(range(num_samples), y_true[:num_samples], label='Actual', alpha=0.7)  # 绘制真实值的散点图
            plt.scatter(range(num_samples), y_pred[:num_samples], label='Predicted', alpha=0.7)  # 绘制预测值的散点图
            plt.ylabel('Class')  # 设置y轴标签
            plt.ylim(-0.5, max(np.max(y_true), np.max(y_pred)) + 0.5)  # 设置y轴范围
        else:  # 如果是回归任务
            # If regression, plot lines
            plt.plot(range(num_samples), y_true[:num_samples], label='Actual', marker='o')  # 绘制真实值的折线图
            plt.plot(range(num_samples), y_pred[:num_samples], label='Predicted', marker='x')  # 绘制预测值的折线图
            plt.ylabel('Value')  # 设置y轴标签
        
        plt.title(f'{title} - Actual vs Predicted Values')  # 设置图表标题
        plt.xlabel('Sample Index')  # 设置x轴标签
        plt.legend()  # 添加图例
        plt.grid(True, alpha=0.3)  # 添加网格线
        plt.tight_layout()  # 优化图表布局
        
        filename = f'plots/{model_name}_predictions.png'  # 设置保存文件名
        plt.savefig(filename)  # 保存图表
        plt.close()  # 关闭图表
        print(f"Predictions plot saved to {filename}")  # 打印保存信息
        
        # 函数功能总结：根据模型类型（LSTM或随机森林）和任务类型（分类或回归）绘制真实值与预测值的对比图，并保存为PNG文件。
    
    def plot_training_history(self):
        """
        Plot LSTM training history
        """
        if 'lstm_history' not in self.results:  # 检查是否有LSTM训练历史
            raise ValueError("No training history found. Train LSTM model first.")  # 如果没有训练历史则抛出错误
        
        history = self.results['lstm_history']  # 获取LSTM训练历史
        
        plt.figure(figsize=(12, 5))  # 创建图表，设置大小
        plt.subplot(1, 2, 1)  # 创建子图1
        plt.plot(history['loss'], label='Training Loss')  # 绘制训练损失
        plt.plot(history['val_loss'], label='Validation Loss')  # 绘制验证损失
        plt.title('LSTM Training and Validation Loss')  # 设置标题
        plt.xlabel('Epoch')  # 设置x轴标签
        plt.ylabel('Loss')  # 设置y轴标签
        plt.legend()  # 添加图例
        plt.grid(True, alpha=0.3)  # 添加网格线
        
        # Plot learning rate if available
        if 'lr' in history:  # 如果历史中包含学习率
            plt.subplot(1, 2, 2)  # 创建子图2
            plt.plot(history['lr'])  # 绘制学习率
            plt.title('Learning Rate')  # 设置标题
            plt.xlabel('Epoch')  # 设置x轴标签
            plt.ylabel('Learning Rate')  # 设置y轴标签
            plt.grid(True, alpha=0.3)  # 添加网格线
        
        plt.tight_layout()  # 优化图表布局
        filename = 'plots/lstm_training_history.png'  # 设置保存文件名
        plt.savefig(filename)  # 保存图表
        plt.close()  # 关闭图表
        print(f"Training history plot saved to {filename}")  # 打印保存信息
        
        # 函数功能总结：绘制LSTM模型的训练历史，包括训练损失、验证损失和学习率（如果可用），并保存为PNG文件。
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance from Random Forest model
        """
        if 'rf_importance' not in self.results:  # 检查是否有随机森林特征重要性
            raise ValueError("No feature importance found. Train Random Forest model first.")  # 如果没有特征重要性则抛出错误
        
        importance = self.results['rf_importance']  # 获取特征重要性
        
        if feature_names is not None:  # 如果提供了特征名称
            # Map feature indices to names
            importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}  # 将特征索引映射到特征名称
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))  # 按重要性降序排序
        
        # Take top 20 features
        features = list(importance.keys())[:20]  # 获取前20个特征名称
        values = list(importance.values())[:20]  # 获取前20个特征重要性值
        
        plt.figure(figsize=(12, 8))  # 创建图表，设置大小
        plt.barh(features, values)  # 绘制水平条形图
        plt.title('Random Forest - Feature Importance')  # 设置标题
        plt.xlabel('Importance')  # 设置x轴标签
        plt.ylabel('Feature')  # 设置y轴标签
        plt.tight_layout()  # 优化图表布局
        
        
        filename = 'plots/rf_feature_importance.png'  # 设置保存文件名
        plt.savefig(filename)  # 保存图表
        plt.close()  # 关闭图表
        print(f"Feature importance plot saved to {filename}")  # 打印保存信息
        
        # 函数功能总结：绘制随机森林模型的特征重要性条形图，显示前20个最重要的特征，并保存为PNG文件。
    
    def compare_models(self, metric='rmse'):
        """
        Compare models based on specified metric
        """
        if 'lstm_metrics' not in self.results or 'rf_metrics' not in self.results:  # 检查是否有两个模型的评估指标
            raise ValueError("Both models must be evaluated before comparison.")  # 如果缺少评估指标则抛出错误
        
        lstm_metrics = self.results['lstm_metrics']  # 获取LSTM模型的评估指标
        rf_metrics = self.results['rf_metrics']  # 获取随机森林模型的评估指标
        
        # Check if metric exists in both results
        if metric not in lstm_metrics or metric not in rf_metrics:  # 检查指定的指标是否存在于两个模型的结果中
            raise ValueError(f"Metric {metric} not found in results.")  # 如果指标不存在则抛出错误
        
        # Compare
        metrics = {  # 创建包含两个模型指标的字典
            'LSTM': lstm_metrics[metric],  # LSTM模型的指标值
            'Random Forest': rf_metrics[metric]  # 随机森林模型的指标值
        }
        
        # For metrics where lower is better (like MSE, RMSE, MAE)
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae']  # 判断指标是否是越低越好
        
        # Plot comparison
        plt.figure(figsize=(10, 6))  # 创建图表，设置大小
        bars = plt.bar(metrics.keys(), metrics.values())  # 绘制条形图
        
        # Color bars based on better/worse
        if lower_is_better:  # 如果指标是越低越好
            best_model = min(metrics, key=metrics.get)  # 找出指标值最低的模型
            for i, (model, _) in enumerate(metrics.items()):  # 遍历模型和指标值
                bars[i].set_color('green' if model == best_model else 'blue')  # 设置颜色，最佳模型为绿色，其他为蓝色
        else:  # 如果指标是越高越好（如R²）
            best_model = max(metrics, key=metrics.get)  # 找出指标值最高的模型
            for i, (model, _) in enumerate(metrics.items()):  # 遍历模型和指标值
                bars[i].set_color('green' if model == best_model else 'blue')  # 设置颜色，最佳模型为绿色，其他为蓝色
        
        plt.title(f'Model Comparison - {metric.upper()}')  # 设置标题
        plt.ylabel(metric.upper())  # 设置y轴标签
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):  # 遍历模型和指标值
            plt.text(i, value + (0.01 * (1 if value >= 0 else -1)), f'{value:.4f}', ha='center')  # 在条形图顶部添加指标值文本
        
        plt.tight_layout()  # 优化图表布局
        filename = f'plots/model_comparison_{metric}.png'  # 设置保存文件名
        plt.savefig(filename)  # 保存图表
        plt.close()  # 关闭图表
        print(f"Model comparison plot saved to {filename}")  # 打印保存信息
        
        return metrics  # 返回指标字典
        
        # 函数功能总结：根据指定的评估指标比较LSTM和随机森林模型的性能，绘制比较条形图，突出显示性能更好的模型，并保存为PNG文件。

# # Example usage
# if __name__ == "__main__":
#     from data_processor import DataProcessor
    
#     # Process data
#     processor = DataProcessor()
#     data = processor.load_data(sample_size=100000)
#     data = processor.preprocess_data()
    
#     # Prepare data for time series forecasting - predicting location
#     X_train, X_test, y_train, y_test, features, scaler = processor.prepare_for_time_series(target_feature='location')
    
#     # Check if we're doing classification or regression
#     is_classification = len(np.unique(y_train)) < 10  # Assume classification if few unique values
#     print(f"Task type: {'Classification' if is_classification else 'Regression'}")
    
#     # Train and evaluate models
#     predictor = TimeSeriesPredictor()
    
#     # Train Random Forest first (it's faster)
#     rf_model = predictor.train_random_forest(X_train, y_train)
#     rf_preds = predictor.predict_random_forest(X_test, y_test)
#     rf_results = predictor.evaluate_random_forest(y_test, rf_preds, is_classification)
    
#     # Train LSTM with a smaller sequence length for demonstration
#     sequence_length = 5
#     lstm_model = predictor.train_lstm(X_train, y_train, sequence_length=sequence_length)
#     lstm_preds, lstm_true = predictor.predict_lstm(X_test, y_test, sequence_length=sequence_length)
#     lstm_results = predictor.evaluate_lstm(lstm_true, lstm_preds, is_classification)
    
#     # Compare models
#     comparison = predictor.compare_models(metric='rmse' if not is_classification else 'mae')
    
#     # Plot predictions
#     predictor.plot_predictions('random_forest', is_classification=is_classification)
#     predictor.plot_predictions('lstm', is_classification=is_classification)
    
#     # Plot feature importance
#     predictor.plot_feature_importance(feature_names=features) 