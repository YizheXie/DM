import pandas as pd  # 导入数据处理库
import numpy as np  # 导入数值计算库
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # 导入数据预处理工具
from sklearn.model_selection import train_test_split  # 导入训练集测试集分割函数
from imblearn.over_sampling import SMOTE  # 导入SMOTE过采样方法
import warnings  # 导入警告控制
import os  # 导入操作系统接口
from datetime import datetime  # 导入日期时间处理

warnings.filterwarnings('ignore')  # 忽略所有警告信息

class DataProcessor:
    def __init__(self, data_path='data/financial_fraud_detection_dataset.csv'):
        """
        Initialize data processor with the path to the dataset
        """
        self.data_path = data_path  # 设置数据文件路径
        self.data = None  # 初始化数据变量
        self.categorical_cols = ['transaction_type', 'merchant_category', 'location',   # 设置分类特征列表
                                'device_used', 'payment_channel']
        self.numerical_cols = ['amount', 'time_since_last_transaction',   # 设置数值特征列表
                              'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
        self.time_col = 'timestamp'  # 设置时间列名
        self.label_col = 'is_fraud'  # 设置标签列名
        # 用于存储类别编码映射关系的字典
        self.class_mappings = {}  # 初始化类别映射字典，用于存储编码映射关系
        
    def load_data(self, sample_size=None):
        """
        Load data from CSV file with optional sampling for development
        """
        print(f"Loading data from {self.data_path}")  # 打印数据加载信息
        # Check if file exists
        if not os.path.exists(self.data_path):  # 检查数据文件是否存在
            raise FileNotFoundError(f"Data file not found at {self.data_path}")  # 如果文件不存在则抛出错误
        
        # Load data with sampling if specified
        if sample_size:  # 如果指定了样本大小
            self.data = pd.read_csv(self.data_path, nrows=sample_size)  # 读取指定数量的样本
            print(f"Loaded sample of {sample_size} rows")  # 打印样本加载信息
        else:
            self.data = pd.read_csv(self.data_path)  # 读取所有数据
            print(f"Loaded all {len(self.data)} rows")  # 打印数据加载信息
            
        # Display basic information
        print(f"Data shape: {self.data.shape}")  # 打印数据形状
        return self.data  # 返回加载的数据
        
        # 函数功能总结：从CSV文件加载数据，可以选择只加载部分样本，并检查文件是否存在，返回加载的数据。
    
    def preprocess_data(self):
        """
        Preprocess the dataset:
        - Handle missing values
        - Convert timestamp to datetime and extract features
        - Encode categorical variables
        """
        if self.data is None:  # 检查数据是否已加载
            raise ValueError("Data not loaded. Call load_data() first.")  # 如果数据未加载则抛出错误
        
        print("Preprocessing data...")  # 打印预处理开始信息
        
        # Handle missing values in numerical columns
        for col in self.numerical_cols:  # 遍历数值特征列
            if col in self.data.columns:  # 如果列存在于数据中
                self.data[col].fillna(self.data[col].median(), inplace=True)  # 使用中位数填充缺失值
        
        # Convert timestamp to datetime and extract features
        if self.time_col in self.data.columns:  # 如果时间列存在于数据中
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])  # 尝试将时间列转换为datetime类型
            except ValueError:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='mixed')  # 如果失败，尝试使用混合格式转换
                
            self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek  # 提取星期几特征
            self.data['hour_of_day'] = self.data['timestamp'].dt.hour  # 提取小时特征
            self.data['month'] = self.data['timestamp'].dt.month  # 提取月份特征
        
        # Encode categorical features
        label_encoders = {}  # 初始化标签编码器字典
        for col in self.categorical_cols:  # 遍历分类特征列
            if col in self.data.columns:  # 如果列存在于数据中
                le = LabelEncoder()  # 创建标签编码器
                self.data[col + '_encoded'] = le.fit_transform(self.data[col])  # 对列进行标签编码并创建新列
                label_encoders[col] = le  # 存储标签编码器
                self.class_mappings[col] = {i: cls for i, cls in enumerate(le.classes_)}  # 存储类别映射关系
        
        print("Preprocessing completed")  # 打印预处理完成信息
        return self.data  # 返回预处理后的数据
        
        # 函数功能总结：对数据进行预处理，包括处理缺失值、转换时间戳并提取时间特征、对分类变量进行编码，并存储类别映射关系。
    
    def create_multiclass_labels(self):
        """
        Create multi-class labels by combining fraud status with device type
        """
        if self.data is None:  # 检查数据是否已加载
            raise ValueError("Data not loaded. Call load_data() first.")  # 如果数据未加载则抛出错误
        
        print("Creating multi-class fraud labels...")  # 打印多分类标签创建开始信息
        
        # Create device-based fraud categories
        self.data['fraud_class'] = 'normal'  # 创建欺诈类别列，默认值为'normal'
        
        # Map different device types for fraudulent transactions
        devices = self.data['device_used'].unique()  # 获取所有设备类型
        for device in devices:  # 遍历每种设备类型
            mask = (self.data['is_fraud'] == True) & (self.data['device_used'] == device)  # 创建掩码，标识使用该设备的欺诈交易
            self.data.loc[mask, 'fraud_class'] = f"{device}_fraud"  # 设置该类别的名称为"{设备}_fraud"
        
        # Create alternative multi-class labels based on payment channel
        self.data['fraud_payment_class'] = 'normal'  # 创建基于支付渠道的欺诈类别列，默认值为'normal'
        channels = self.data['payment_channel'].unique()  # 获取所有支付渠道
        for channel in channels:  # 遍历每种支付渠道
            mask = (self.data['is_fraud'] == True) & (self.data['payment_channel'] == channel)  # 创建掩码，标识使用该支付渠道的欺诈交易
            self.data.loc[mask, 'fraud_payment_class'] = f"{channel}_fraud"  # 设置该类别的名称为"{支付渠道}_fraud"
        
        # Encode multi-class labels
        le_fraud_class = LabelEncoder()  # 创建欺诈类别的标签编码器
        self.data['fraud_class_encoded'] = le_fraud_class.fit_transform(self.data['fraud_class'])  # 对欺诈类别进行标签编码
        
        self.class_mappings['fraud_class'] = {i: cls for i, cls in enumerate(le_fraud_class.classes_)}  # 存储欺诈类别的映射关系
        
        le_payment_class = LabelEncoder()  # 创建支付渠道欺诈类别的标签编码器
        self.data['fraud_payment_class_encoded'] = le_payment_class.fit_transform(self.data['fraud_payment_class'])  # 对支付渠道欺诈类别进行标签编码
        
        self.class_mappings['fraud_payment_class'] = {i: cls for i, cls in enumerate(le_payment_class.classes_)}  # 存储支付渠道欺诈类别的映射关系
        
        print("Fraud class mapping:")  # 打印欺诈类别映射信息
        for code, name in self.class_mappings['fraud_class'].items():  # 遍历欺诈类别映射关系
            print(f"  {code} -> {name}")  # 打印编码和类别名称
        
        # Display class distribution
        print("Fraud class distribution:")  # 打印欺诈类别分布信息
        print(self.data['fraud_class'].value_counts(normalize=True) * 100)  # 打印欺诈类别的百分比分布
        
        return self.data  # 返回添加了多分类标签的数据
        
        # 函数功能总结：创建多分类欺诈标签，包括基于设备类型和支付渠道的欺诈分类，对这些类别进行编码，并存储类别映射关系。
    
    def handle_imbalance(self, target_col='fraud_class_encoded', strategy='smote'):
        """
        Handle class imbalance using various strategies
        """
        if self.data is None:  # 检查数据是否已加载
            raise ValueError("Data not loaded. Call load_data() first.")  # 如果数据未加载则抛出错误
        
        print(f"Handling class imbalance using {strategy}...")  # 打印处理类别不平衡开始信息
        
        # Select features and target
        features = []  # 初始化特征列表
        
        # Add encoded categorical features
        for col in self.categorical_cols:  # 遍历分类特征列
            if col + '_encoded' in self.data.columns:  # 如果编码后的列存在于数据中
                features.append(col + '_encoded')  # 将编码后的列添加到特征列表
        
        # Add numerical features
        for col in self.numerical_cols:  # 遍历数值特征列
            if col in self.data.columns:  # 如果列存在于数据中
                features.append(col)  # 将列添加到特征列表
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:  # 如果星期几特征存在于数据中
            features.extend(['day_of_week', 'hour_of_day', 'month'])  # 将时间特征添加到特征列表
        
        X = self.data[features]  # 提取特征数据
        y = self.data[target_col]  # 提取目标变量
        
        # Split data before applying SMOTE to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(  # 分割训练集和测试集
            X, y, test_size=0.2, random_state=42, stratify=y  # 设置测试集大小、随机种子和分层抽样
        )
        
        # Apply SMOTE to training data only
        if strategy.lower() == 'smote':  # 如果使用SMOTE策略
            smote = SMOTE(random_state=42)  # 创建SMOTE对象
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # 对训练数据进行SMOTE过采样
            print(f"Original training class distribution: {pd.Series(y_train).value_counts()}")  # 打印原始训练集类别分布
            print(f"Resampled training class distribution: {pd.Series(y_train_resampled).value_counts()}")  # 打印重采样后的训练集类别分布
            
            target_prefix = target_col.replace('_encoded', '')  # 获取目标列的前缀
            if target_prefix in self.class_mappings:  # 如果目标列的映射关系存在
                print("Class distribution with names:")  # 打印带名称的类别分布
                for class_code, count in pd.Series(y_train_resampled).value_counts().items():  # 遍历重采样后的类别分布
                    class_name = self.class_mappings[target_prefix][class_code]  # 获取类别名称
                    print(f"  {class_code} ({class_name}): {count}")  # 打印类别编码、名称和数量
            
            return X_train_resampled, X_test, y_train_resampled, y_test, features  # 返回重采样后的训练数据、测试数据和特征列表
        
        # No resampling
        return X_train, X_test, y_train, y_test, features  # 如果不使用重采样，直接返回分割后的数据和特征列表
        
        # 函数功能总结：处理类别不平衡问题，包括特征选择、数据分割，并在训练数据上应用SMOTE过采样，最后返回处理后的数据集和特征列表。
    
    def prepare_for_binary_anomaly_detection(self):
        """
        Prepare data for binary anomaly detection
        """
        if self.data is None:  # 检查数据是否已加载
            raise ValueError("Data not loaded. Call load_data() first.")  # 如果数据未加载则抛出错误
        
        print("Preparing data for anomaly detection...")  # 打印异常检测数据准备开始信息
        
        # Select relevant features
        features = []  # 初始化特征列表
        
        # Add encoded categorical features
        for col in self.categorical_cols:  # 遍历分类特征列
            if col + '_encoded' in self.data.columns:  # 如果编码后的列存在于数据中
                features.append(col + '_encoded')  # 将编码后的列添加到特征列表
        
        # Add numerical features
        for col in self.numerical_cols:  # 遍历数值特征列
            if col in self.data.columns:  # 如果列存在于数据中
                features.append(col)  # 将列添加到特征列表
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:  # 如果星期几特征存在于数据中
            features.extend(['day_of_week', 'hour_of_day', 'month'])  # 将时间特征添加到特征列表
        
        X = self.data[features]  # 提取特征数据
        y = self.data[self.label_col].astype(int)  # 提取目标变量并转换为整数类型
        
        # Standardize features
        scaler = StandardScaler()  # 创建标准化器
        X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(  # 分割训练集和测试集
            X_scaled, y, test_size=0.2, random_state=42, stratify=y  # 设置测试集大小、随机种子和分层抽样
        )
        
        return X_train, X_test, y_train, y_test, features, scaler  # 返回训练数据、测试数据、特征列表和标准化器
        
        # 函数功能总结：准备用于二分类异常检测的数据，包括特征选择、标准化和数据分割，最后返回处理后的数据集、特征列表和标准化器。
    
    def prepare_for_time_series(self, target_feature='location'):
        """
        Prepare data for time series prediction
        Uses timestamp to create sequence-based features
        Target can be one of the metadata features (not IP or hash)
        """
        if self.data is None:  # 检查数据是否已加载
            raise ValueError("Data not loaded. Call load_data() first.")  # 如果数据未加载则抛出错误
        
        print(f"Preparing data for time series prediction of '{target_feature}'...")  # 打印时间序列预测数据准备开始信息
        
        # Sort data by timestamp
        if 'timestamp' in self.data.columns:  # 如果时间戳列存在于数据中
            self.data = self.data.sort_values('timestamp')  # 按时间戳排序数据
        
        # Select features excluding IP address and device hash
        features = []  # 初始化特征列表
        
        # Add encoded categorical features except the target
        for col in self.categorical_cols:  # 遍历分类特征列
            if col != target_feature and col + '_encoded' in self.data.columns:  # 如果列不是目标特征且编码后的列存在于数据中
                features.append(col + '_encoded')  # 将编码后的列添加到特征列表
        
        # Add numerical features
        for col in self.numerical_cols:  # 遍历数值特征列
            if col in self.data.columns:  # 如果列存在于数据中
                features.append(col)  # 将列添加到特征列表
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:  # 如果星期几特征存在于数据中
            features.extend(['day_of_week', 'hour_of_day', 'month'])  # 将时间特征添加到特征列表
        
        X = self.data[features]  # 提取特征数据
        
        # Prepare target
        if target_feature + '_encoded' in self.data.columns:  # 如果目标特征的编码列存在于数据中
            y = self.data[target_feature + '_encoded']  # 使用编码后的目标列
        else:
            le = LabelEncoder()  # 创建标签编码器
            y = le.fit_transform(self.data[target_feature])  # 对目标特征进行标签编码
            # 保存编码映射关系
            self.class_mappings[target_feature] = {i: cls for i, cls in enumerate(le.classes_)}  # 存储目标特征的映射关系
        
        # Standardize features
        scaler = StandardScaler()  # 创建标准化器
        X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化
        
        # Split data while preserving time order
        train_size = int(len(X_scaled) * 0.8)  # 计算训练集大小（80%的数据）
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]  # 按时间顺序分割特征数据
        y_train, y_test = y[:train_size], y[train_size:]  # 按时间顺序分割目标变量
        
        return X_train, X_test, y_train, y_test, features, scaler  # 返回训练数据、测试数据、特征列表和标准化器
        
        # 函数功能总结：准备用于时间序列预测的数据，包括按时间戳排序、特征选择、目标变量编码、标准化，并按时间顺序分割数据，最后返回处理后的数据集、特征列表和标准化器。
    
    def get_class_names(self, target_col='fraud_class_encoded'):
        """
        Get the sorted class names list in order of encoding
        """
        target_prefix = target_col.replace('_encoded', '')  # 获取目标列的前缀
        if target_prefix in self.class_mappings:  # 如果目标列的映射关系存在
            mapping = self.class_mappings[target_prefix]  # 获取映射关系
            # return the sorted class names list
            return [mapping[i] for i in range(len(mapping))]  # 返回按编码顺序排序的类别名称列表
        return None  # 如果映射关系不存在则返回None
        
        # 函数功能总结：根据目标列的编码获取按编码顺序排序的类别名称列表，如果映射关系不存在则返回None。

# Example usage
# if __name__ == "__main__":
#     processor = DataProcessor()
#     # Use a small sample for testing
#     data = processor.load_data(sample_size=100000)
#     data = processor.preprocess_data()
#     data = processor.create_multiclass_labels()
    
#     # Prepare data for classification
#     X_train, X_test, y_train, y_test, features = processor.handle_imbalance()
#     print(f"Classification data prepared with {len(features)} features")
    
#     # Get proper class names
#     class_names = processor.get_class_names('fraud_class_encoded')
#     print(f"Class names in order: {class_names}")
    
#     # Prepare data for anomaly detection
#     X_train_anom, X_test_anom, y_train_anom, y_test_anom, features_anom, scaler_anom = processor.prepare_for_binary_anomaly_detection()
#     print(f"Anomaly detection data prepared with {len(features_anom)} features")
    
#     # Prepare data for time series prediction
#     X_train_ts, X_test_ts, y_train_ts, y_test_ts, features_ts, scaler_ts = processor.prepare_for_time_series(target_feature='location')
#     print(f"Time series data prepared with {len(features_ts)} features") 