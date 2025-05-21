import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库
import matplotlib  # 导入绘图基础库
matplotlib.use('Agg')  # 设置matplotlib后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt  # 导入matplotlib的绘图功能
import seaborn as sns  # 导入高级绘图库
from sklearn.cluster import KMeans  # 导入K均值聚类算法
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report  # 导入评估指标
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # 导入分类评估指标
from sklearn.preprocessing import StandardScaler  # 导入标准化工具
from sklearn.decomposition import PCA  # 导入主成分分析
import tensorflow as tf  # 导入TensorFlow
from tensorflow.keras.models import Model  # 导入Keras模型
from tensorflow.keras.layers import Input, Dense, Dropout  # 导入Keras层
from tensorflow.keras.callbacks import EarlyStopping  # 导入早停回调
import time  # 导入时间库用于计时
import warnings  # 导入警告控制
import os  # 导入操作系统接口
os.makedirs('plots', exist_ok=True)  # 创建plots文件夹，如果已存在则不报错

warnings.filterwarnings('ignore')  # 忽略所有警告信息

class AnomalyDetector:
    def __init__(self):
        """
        Initialize anomaly detector with autoencoder and KMeans models
        """
        self.models = {}  # 初始化存储模型的字典
        self.results = {}  # 初始化存储结果的字典
        self.thresholds = {}  # 初始化存储阈值的字典
    
    def train_autoencoder(self, X_train, y_train=None, hidden_dims=[64, 32, 16], 
                          activation='relu', epochs=50, batch_size=256):
        """
        Train deep learning autoencoder for anomaly detection
        """
        print("Training Autoencoder for anomaly detection...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Get input dimensions
        input_dim = X_train.shape[1]  # 获取输入特征维度
        
        # Create encoder layers
        input_layer = Input(shape=(input_dim,))  # 创建输入层
        
        # Build encoder
        encoder = input_layer  # 设置编码器输入
        for dim in hidden_dims:  # 遍历隐藏层维度
            encoder = Dense(dim, activation=activation)(encoder)  # 添加全连接层
            encoder = Dropout(0.2)(encoder)  # 添加Dropout层防止过拟合
        
        # Build decoder (symmetrical to encoder)
        decoder = encoder  # 设置解码器输入
        for dim in reversed(hidden_dims[:-1]):  # 遍历反转的隐藏层维度（除了最后一个）
            decoder = Dense(dim, activation=activation)(decoder)  # 添加全连接层
            decoder = Dropout(0.2)(decoder)  # 添加Dropout层
        
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoder)  # 创建输出层，维度与输入相同
        
        # Create model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)  # 创建自编码器模型
        
        # Compile model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # 使用Adam优化器和均方误差损失函数
        
        # Print model summary
        print(autoencoder.summary())  # 打印模型结构
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # 创建早停回调
        
        history = autoencoder.fit(  # 训练模型
            X_train, X_train,  # 自编码器输入和目标都是X_train
            epochs=epochs,  # 训练轮数
            batch_size=batch_size,  # 批次大小
            shuffle=True,  # 打乱数据
            validation_split=0.1,  # 使用10%的数据作为验证集
            callbacks=[early_stopping],  # 使用早停回调
            verbose=1  # 显示训练进度
        )
        
        # Store the model
        self.models['autoencoder'] = autoencoder  # 存储训练好的自编码器模型
        
        # Calculate reconstruction error
        reconstructions = autoencoder.predict(X_train)  # 重建训练数据
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)  # 计算每个样本的重建误差
        
        # If true labels are provided, calculate threshold using precision-recall trade-off
        if y_train is not None:  # 如果提供了训练集标签
            # Find a good threshold based on F1 score
            thresholds = np.linspace(np.min(mse), np.max(mse), 100)  # 创建100个均匀分布的阈值候选
            best_f1 = 0  # 初始化最佳F1分数
            best_threshold = 0  # 初始化最佳阈值
            
            for threshold in thresholds:  # 遍历每个阈值候选
                y_pred = (mse > threshold).astype(int)  # 根据阈值将重建误差转换为预测标签
                if np.sum(y_pred) > 0 and np.sum(y_train) > 0:  # 如果预测和真实都有正样本
                    f1 = f1_score(y_train, y_pred)  # 计算F1分数
                    if f1 > best_f1:  # 如果F1分数更好
                        best_f1 = f1  # 更新最佳F1分数
                        best_threshold = threshold  # 更新最佳阈值
            
            print(f"Best threshold: {best_threshold:.6f} with F1: {best_f1:.4f}")  # 打印最佳阈值和F1分数
            self.thresholds['autoencoder'] = best_threshold  # 存储自编码器的最佳阈值
        else:
            # Use simple statistical approach: mean + std
            threshold = np.mean(mse) + np.std(mse)  # 使用均值加一个标准差作为阈值
            self.thresholds['autoencoder'] = threshold  # 存储统计阈值
            print(f"Statistical threshold: {threshold:.6f}")  # 打印统计阈值
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"Autoencoder training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        # Store training history
        self.results['autoencoder_history'] = history.history  # 存储训练历史
        
        return autoencoder  # 返回训练好的自编码器模型
        
        # 函数功能总结：训练深度学习自编码器模型用于异常检测，计算重建误差，确定最佳阈值，并存储模型、阈值和训练历史。
    
    def train_kmeans(self, X_train, y_train=None, n_clusters=2):
        """
        Train KMeans for anomaly detection
        """
        print("Training KMeans for anomaly detection...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # 创建K均值聚类模型
        kmeans.fit(X_train)  # 训练K均值聚类模型
        
        # Store the model
        self.models['kmeans'] = kmeans  # 存储训练好的K均值聚类模型
        
        # Calculate cluster centers and distances
        cluster_centers = kmeans.cluster_centers_  # 获取聚类中心
        cluster_labels = kmeans.labels_  # 获取样本的聚类标签
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:  # 如果聚类数大于1
            silhouette = silhouette_score(X_train, cluster_labels)  # 计算轮廓系数
            print(f"Silhouette Score: {silhouette:.4f}")  # 打印轮廓系数
        
        # Calculate distance to cluster center for each point
        distances = np.zeros(X_train.shape[0])  # 初始化距离数组
        for i in range(X_train.shape[0]):  # 遍历每个样本
            cluster_idx = cluster_labels[i]  # 获取样本的聚类标签
            distances[i] = np.linalg.norm(X_train[i] - cluster_centers[cluster_idx])  # 计算样本到其聚类中心的距离
        
        # If true labels are provided, determine which cluster is the fraud cluster
        if y_train is not None:  # 如果提供了训练集标签
            # Check each cluster for fraud ratio
            fraud_ratios = {}  # 初始化欺诈比例字典
            for cluster in range(n_clusters):  # 遍历每个聚类
                cluster_samples = (cluster_labels == cluster)  # 获取属于当前聚类的样本
                if np.sum(cluster_samples) > 0:  # 如果有样本属于当前聚类
                    fraud_ratio = np.mean(y_train[cluster_samples])  # 计算当前聚类中的欺诈比例
                    fraud_ratios[cluster] = fraud_ratio  # 存储欺诈比例
                    print(f"Cluster {cluster}: {np.sum(cluster_samples)} samples, fraud ratio: {fraud_ratio:.4f}")  # 打印聚类信息
            
            # Identify the fraud cluster (highest fraud ratio)
            fraud_cluster = max(fraud_ratios, key=fraud_ratios.get)  # 获取欺诈比例最高的聚类
            print(f"Identified fraud cluster: {fraud_cluster}")  # 打印识别出的欺诈聚类
            
            # Store fraud cluster
            self.results['fraud_cluster'] = fraud_cluster  # 存储欺诈聚类
            
            # Define distance threshold based on this cluster
            cluster_points = X_train[cluster_labels == fraud_cluster]  # 获取欺诈聚类中的样本
            if len(cluster_points) > 0:  # 如果欺诈聚类中有样本
                threshold = np.percentile(  # 计算基于欺诈聚类的距离阈值
                    np.linalg.norm(cluster_points - cluster_centers[fraud_cluster], axis=1),  # 计算欺诈聚类中样本到聚类中心的距离
                    75  # 使用75%分位数作为阈值
                )
                self.thresholds['kmeans'] = threshold  # 存储K均值聚类的阈值
                print(f"Distance threshold for fraud cluster: {threshold:.6f}")  # 打印距离阈值
        else:
            # Determine fraud cluster based on distances (assume fraud points are further from their centers)
            mean_distances = []  # 初始化平均距离列表
            for cluster in range(n_clusters):  # 遍历每个聚类
                cluster_samples = (cluster_labels == cluster)  # 获取属于当前聚类的样本
                if np.sum(cluster_samples) > 0:  # 如果有样本属于当前聚类
                    mean_dist = np.mean(distances[cluster_samples])  # 计算当前聚类中样本的平均距离
                    mean_distances.append((cluster, mean_dist))  # 添加聚类索引和平均距离
            
            # Cluster with highest average distance is likely the fraud cluster
            fraud_cluster = max(mean_distances, key=lambda x: x[1])[0]  # 获取平均距离最大的聚类
            print(f"Assumed fraud cluster (based on distances): {fraud_cluster}")  # 打印假设的欺诈聚类
            
            # Store fraud cluster
            self.results['fraud_cluster'] = fraud_cluster  # 存储欺诈聚类
            
            # Define distance threshold
            threshold = np.percentile(distances, 70)  # 使用距离的70%分位数作为阈值
            self.thresholds['kmeans'] = threshold  # 存储K均值聚类的阈值
            print(f"Distance threshold (statistical): {threshold:.6f}")  # 打印统计阈值
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"KMeans training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        return kmeans  # 返回训练好的K均值聚类模型
        
        # 函数功能总结：训练K均值聚类模型用于异常检测，确定欺诈聚类，计算样本到聚类中心的距离，设定距离阈值，并存储模型、欺诈聚类和阈值。
    
    def detect_anomalies_autoencoder(self, X_test, threshold=None):
        """
        Detect anomalies using the trained autoencoder
        """
        if 'autoencoder' not in self.models:  # 检查自编码器模型是否已训练
            raise ValueError("Autoencoder model not found. Train the model first.")  # 如果未训练则抛出错误
        
        # Use stored threshold if not provided
        if threshold is None:  # 如果没有提供阈值
            threshold = self.thresholds.get('autoencoder')  # 获取存储的自编码器阈值
            if threshold is None:  # 如果没有存储的阈值
                raise ValueError("No threshold found for autoencoder. Train with labels or provide threshold.")  # 抛出错误
        
        # Get model
        autoencoder = self.models['autoencoder']  # 获取训练好的自编码器模型
        
        # Calculate reconstruction error
        reconstructions = autoencoder.predict(X_test)  # 重建测试数据
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)  # 计算每个样本的重建误差
        
        # Classify as anomaly if error > threshold
        anomalies = (mse > threshold).astype(int)  # 根据阈值将重建误差转换为异常标签
        
        # Store reconstruction errors
        self.results['autoencoder_errors'] = mse  # 存储重建误差
        
        return anomalies, mse  # 返回异常标签和重建误差
        
        # 函数功能总结：使用训练好的自编码器模型检测测试数据中的异常，计算重建误差，根据阈值确定异常样本，并返回异常标签和重建误差。
    
    def detect_anomalies_kmeans(self, X_test, threshold=None):
        """
        Detect anomalies using the trained KMeans model
        """
        if 'kmeans' not in self.models:  # 检查K均值聚类模型是否已训练
            raise ValueError("KMeans model not found. Train the model first.")  # 如果未训练则抛出错误
        
        # Use stored threshold if not provided
        if threshold is None:  # 如果没有提供阈值
            threshold = self.thresholds.get('kmeans')  # 获取存储的K均值聚类阈值
            if threshold is None:  # 如果没有存储的阈值
                raise ValueError("No threshold found for KMeans. Train with labels or provide threshold.")  # 抛出错误
        
        # Get model and fraud cluster
        kmeans = self.models['kmeans']  # 获取训练好的K均值聚类模型
        fraud_cluster = self.results.get('fraud_cluster')  # 获取欺诈聚类
        
        if fraud_cluster is None:  # 如果没有确定欺诈聚类
            raise ValueError("Fraud cluster not identified. Train with labels first.")  # 抛出错误
        
        # Predict clusters
        cluster_labels = kmeans.predict(X_test)  # 预测测试数据的聚类标签
        
        # Calculate distances to assigned cluster centers
        distances = np.zeros(X_test.shape[0])  # 初始化距离数组
        for i in range(X_test.shape[0]):  # 遍历每个样本
            cluster_idx = cluster_labels[i]  # 获取样本的聚类标签
            distances[i] = np.linalg.norm(X_test[i] - kmeans.cluster_centers_[cluster_idx])  # 计算样本到其聚类中心的距离
        
        # Method 1: Points in fraud cluster with distance > threshold
        anomalies_1 = ((cluster_labels == fraud_cluster) & (distances > threshold)).astype(int)  # 欺诈聚类中距离大于阈值的样本标记为异常
        
        # Method 2: Points in fraud cluster
        anomalies_2 = (cluster_labels == fraud_cluster).astype(int)  # 所有属于欺诈聚类的样本标记为异常
        
        # Store the distances
        self.results['kmeans_distances'] = distances  # 存储距离
        self.results['kmeans_clusters'] = cluster_labels  # 存储聚类标签
        
        # Return the better method based on previous validation
        return anomalies_2, distances  # 返回方法2的异常标签和距离
        
        # 函数功能总结：使用训练好的K均值聚类模型检测测试数据中的异常，计算样本到聚类中心的距离，根据欺诈聚类和距离阈值确定异常样本，并返回异常标签和距离。
    
    def evaluate_detector(self, method, y_pred, y_true, scores=None):
        """
        Evaluate anomaly detection performance
        """
        print(f"Evaluating {method} anomaly detection...")  # 打印评估开始的信息
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)  # 计算准确率
        precision = precision_score(y_true, y_pred, zero_division=0)  # 计算精确率，避免除零错误
        recall = recall_score(y_true, y_pred)  # 计算召回率
        f1 = f1_score(y_true, y_pred)  # 计算F1分数
        
        # Store results
        self.results[f'{method}_metrics'] = {  # 创建评估结果字典
            'accuracy': accuracy,  # 存储准确率
            'precision': precision,  # 存储精确率
            'recall': recall,  # 存储召回率
            'f1': f1,  # 存储F1分数
            'confusion_matrix': confusion_matrix(y_true, y_pred),  # 存储混淆矩阵
            'y_true': y_true,  # 存储真实标签
            'y_pred': y_pred  # 存储预测标签
        }
        
        if scores is not None:  # 如果提供了分数
            self.results[f'{method}_scores'] = scores  # 存储分数
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")  # 打印准确率
        print(f"Precision: {precision:.4f}")  # 打印精确率
        print(f"Recall: {recall:.4f}")  # 打印召回率
        print(f"F1 Score: {f1:.4f}")  # 打印F1分数
        
        print("\nConfusion Matrix:")  # 打印混淆矩阵标题
        print(confusion_matrix(y_true, y_pred))  # 打印混淆矩阵
        
        print("\nClassification Report:")  # 打印分类报告标题
        print(classification_report(y_true, y_pred))  # 打印分类报告
        
        return self.results[f'{method}_metrics']  # 返回评估结果
        
        # 函数功能总结：评估异常检测模型的性能，计算准确率、精确率、召回率和F1分数等指标，打印混淆矩阵和分类报告，并返回评估结果。
    
    def plot_reconstruction_error(self, method='autoencoder', y_true=None):
        """
        Plot reconstruction error distribution for autoencoder
        """
        if method == 'autoencoder':  # 如果是自编码器方法
            errors = self.results.get('autoencoder_errors')  # 获取自编码器的重建误差
            if errors is None:  # 如果重建误差不存在
                raise ValueError("No reconstruction errors found. Run detect_anomalies_autoencoder first.")  # 抛出错误
            
            threshold = self.thresholds.get('autoencoder')  # 获取自编码器的阈值
            title = 'Autoencoder Reconstruction Error'  # 设置图表标题
            xlabel = 'Reconstruction Error (MSE)'  # 设置x轴标签
        
        elif method == 'kmeans':  # 如果是K均值聚类方法
            errors = self.results.get('kmeans_distances')  # 获取K均值聚类的距离
            if errors is None:  # 如果距离不存在
                raise ValueError("No distances found. Run detect_anomalies_kmeans first.")  # 抛出错误
            
            threshold = self.thresholds.get('kmeans')  # 获取K均值聚类的阈值
            title = 'KMeans Distance to Cluster Center'  # 设置图表标题
            xlabel = 'Distance'  # 设置x轴标签
        
        else:
            raise ValueError(f"Unsupported method: {method}")  # 如果方法不支持则抛出错误
        
        plt.figure(figsize=(12, 6))  # 创建图表，设置大小
        
        if y_true is not None:  # 如果提供了真实标签
            # Plot distribution by class
            plt.hist(errors[y_true == 0], bins=50, alpha=0.5, color='blue', label='Normal')  # 绘制正常样本的误差直方图
            plt.hist(errors[y_true == 1], bins=50, alpha=0.5, color='red', label='Fraud')  # 绘制欺诈样本的误差直方图
            plt.legend()  # 添加图例
        else:
            # Plot overall distribution
            plt.hist(errors, bins=50, alpha=0.7)  # 绘制整体误差分布直方图
        
        if threshold is not None:  # 如果提供了阈值
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')  # 绘制阈值线
            plt.legend()  # 添加图例
        
        plt.title(title)  # 设置标题
        plt.xlabel(xlabel)  # 设置x轴标签
        plt.ylabel('Count')  # 设置y轴标签
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'{method}_reconstruction_error.png'))  # 保存图表
        
        # 函数功能总结：绘制自编码器的重建误差或K均值聚类的距离分布直方图，如果提供真实标签则按类别分别绘制，并标记阈值线，最后保存为PNG文件。
    
    def plot_clusters(self, X, method='kmeans', y_true=None):
        """
        Plot clusters in 2D space (using PCA reduction)
        """
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)  # 创建PCA对象，降至2维
        X_2d = pca.fit_transform(X)  # 对数据进行PCA降维
        
        plt.figure(figsize=(12, 10))  # 创建图表，设置大小
        
        if method == 'kmeans':  # 如果是K均值聚类方法
            if 'kmeans_clusters' not in self.results:  # 如果聚类标签不存在
                raise ValueError("No cluster assignments found. Run detect_anomalies_kmeans first.")  # 抛出错误
            
            cluster_labels = self.results['kmeans_clusters']  # 获取聚类标签
            
            # Plot each cluster
            for cluster in np.unique(cluster_labels):  # 遍历每个聚类
                cluster_points = X_2d[cluster_labels == cluster]  # 获取属于当前聚类的样本
                plt.scatter(  # 绘制散点图
                    cluster_points[:, 0],  # x坐标
                    cluster_points[:, 1],  # y坐标
                    label=f'Cluster {cluster}',  # 标签
                    alpha=0.6  # 透明度
                )
            
            # Plot cluster centers
            kmeans = self.models['kmeans']  # 获取K均值聚类模型
            centers_2d = pca.transform(kmeans.cluster_centers_)  # 对聚类中心进行PCA降维
            plt.scatter(  # 绘制聚类中心
                centers_2d[:, 0],  # x坐标
                centers_2d[:, 1],  # y坐标
                s=200,  # 点大小
                marker='X',  # 点形状
                c='black',  # 点颜色
                label='Centroids'  # 标签
            )
            
            title = 'KMeans Clusters (PCA 2D projection)'  # 设置图表标题
        
        elif method == 'autoencoder':  # 如果是自编码器方法
            if 'autoencoder_errors' not in self.results:  # 如果重建误差不存在
                raise ValueError("No reconstruction errors found. Run detect_anomalies_autoencoder first.")  # 抛出错误
            
            errors = self.results['autoencoder_errors']  # 获取重建误差
            threshold = self.thresholds.get('autoencoder')  # 获取阈值
            
            # Color by reconstruction error
            sc = plt.scatter(  # 绘制散点图
                X_2d[:, 0],  # x坐标
                X_2d[:, 1],  # y坐标
                c=errors,  # 颜色根据误差值
                cmap='viridis',  # 使用viridis颜色映射
                alpha=0.6  # 透明度
            )
            plt.colorbar(sc, label='Reconstruction Error')  # 添加颜色条
            
            if threshold is not None:  # 如果提供了阈值
                # Mark points above threshold
                anomalies = errors > threshold  # 获取超过阈值的样本
                plt.scatter(  # 绘制异常点
                    X_2d[anomalies, 0],  # x坐标
                    X_2d[anomalies, 1],  # y坐标
                    s=50,  # 点大小
                    edgecolors='red',  # 边缘颜色
                    facecolors='none',  # 填充颜色为空（仅显示边缘）
                    linewidths=2,  # 边缘宽度
                    label='Detected Anomalies'  # 标签
                )
            
            title = 'Autoencoder Anomalies (PCA 2D projection)'  # 设置图表标题
        
        else:
            raise ValueError(f"Unsupported method: {method}")  # 如果方法不支持则抛出错误
        
        # If true labels provided, add contour
        if y_true is not None:  # 如果提供了真实标签
            plt.scatter(  # 绘制真实欺诈样本
                X_2d[y_true == 1, 0],  # x坐标
                X_2d[y_true == 1, 1],  # y坐标
                s=50,  # 点大小
                edgecolors='red',  # 边缘颜色
                facecolors='none',  # 填充颜色为空（仅显示边缘）
                linewidths=2,  # 边缘宽度
                label='True Fraud'  # 标签
            )
        
        plt.title(title)  # 设置标题
        plt.xlabel('PCA Component 1')  # 设置x轴标签
        plt.ylabel('PCA Component 2')  # 设置y轴标签
        plt.legend()  # 添加图例
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'{method}_clusters.png'))  # 保存图表
        
        # 函数功能总结：使用PCA将数据降至2维并绘制可视化图，对于K均值聚类方法显示各个聚类和聚类中心，对于自编码器方法根据重建误差着色并标记异常点，如果提供真实标签则标记真实欺诈样本，最后保存为PNG文件。
    
    def compare_methods(self, metric='f1'):
        """
        Compare both anomaly detection methods
        """
        if f'autoencoder_metrics' not in self.results or f'kmeans_metrics' not in self.results:  # 如果任一方法的评估指标不存在
            raise ValueError("Both methods must be evaluated before comparison.")  # 抛出错误
        
        # Get metrics
        autoencoder_metrics = self.results['autoencoder_metrics']  # 获取自编码器的评估指标
        kmeans_metrics = self.results['kmeans_metrics']  # 获取K均值聚类的评估指标
        
        # Compare
        metrics = {  # 创建指标对比字典
            'Autoencoder': autoencoder_metrics[metric],  # 自编码器的指定指标值
            'KMeans': kmeans_metrics[metric]  # K均值聚类的指定指标值
        }
        
        # Plot comparison
        plt.figure(figsize=(10, 6))  # 创建图表，设置大小
        plt.bar(metrics.keys(), metrics.values())  # 绘制条形图
        plt.title(f'Anomaly Detection Methods Comparison - {metric}')  # 设置标题
        plt.ylabel(metric)  # 设置y轴标签
        plt.ylim(0, 1)  # 设置y轴范围
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):  # 遍历每个模型及其指标值
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center')  # 在条形图顶部添加文本
        
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', 'comparison.png'))  # 保存图表
        
        return metrics  # 返回指标对比字典
        
        # 函数功能总结：比较自编码器和K均值聚类两种异常检测方法的性能（基于指定的评估指标），绘制条形图，并保存为PNG文件。

# # Example usage
# if __name__ == "__main__":
#     from data_processor import DataProcessor
    
#     # Process data
#     processor = DataProcessor()
#     data = processor.load_data(sample_size=100000)
#     data = processor.preprocess_data()
    
#     # Prepare data for anomaly detection
#     X_train, X_test, y_train, y_test, features, scaler = processor.prepare_for_binary_anomaly_detection()
    
#     # Train and evaluate models
#     detector = AnomalyDetector()
    
#     # Train autoencoder
#     autoencoder = detector.train_autoencoder(X_train, y_train)
#     anomalies_ae, scores_ae = detector.detect_anomalies_autoencoder(X_test)
#     ae_results = detector.evaluate_detector('autoencoder', anomalies_ae, y_test, scores_ae)
    
#     # Train KMeans
#     kmeans = detector.train_kmeans(X_train, y_train, n_clusters=2)
#     anomalies_km, scores_km = detector.detect_anomalies_kmeans(X_test)
#     km_results = detector.evaluate_detector('kmeans', anomalies_km, y_test, scores_km)
    
#     # Compare methods
#     comparison = detector.compare_methods(metric='f1')
    
#     # Plot reconstruction error
#     detector.plot_reconstruction_error('autoencoder', y_test)
    
#     # Plot clusters
#     detector.plot_clusters(X_test, 'kmeans', y_test) 