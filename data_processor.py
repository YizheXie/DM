import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path='data/financial_fraud_detection_dataset.csv'):
        """
        Initialize data processor with the path to the dataset
        """
        self.data_path = data_path
        self.data = None
        self.categorical_cols = ['transaction_type', 'merchant_category', 'location', 
                                'device_used', 'payment_channel']
        self.numerical_cols = ['amount', 'time_since_last_transaction', 
                              'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
        self.time_col = 'timestamp'
        self.label_col = 'is_fraud'
        # 用于存储类别编码映射关系的字典
        self.class_mappings = {}
        
    def load_data(self, sample_size=None):
        """
        Load data from CSV file with optional sampling for development
        """
        print(f"Loading data from {self.data_path}")
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        # Load data with sampling if specified
        if sample_size:
            self.data = pd.read_csv(self.data_path, nrows=sample_size)
            print(f"Loaded sample of {sample_size} rows")
        else:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded all {len(self.data)} rows")
            
        # Display basic information
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the dataset:
        - Handle missing values
        - Convert timestamp to datetime and extract features
        - Encode categorical variables
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preprocessing data...")
        
        # Handle missing values in numerical columns
        for col in self.numerical_cols:
            if col in self.data.columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Convert timestamp to datetime and extract features
        if self.time_col in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            except ValueError:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='mixed')
                
            self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
            self.data['hour_of_day'] = self.data['timestamp'].dt.hour
            self.data['month'] = self.data['timestamp'].dt.month
        
        # Encode categorical features
        label_encoders = {}
        for col in self.categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col + '_encoded'] = le.fit_transform(self.data[col])
                label_encoders[col] = le
                self.class_mappings[col] = {i: cls for i, cls in enumerate(le.classes_)}
        
        print("Preprocessing completed")
        return self.data
    
    def create_multiclass_labels(self):
        """
        Create multi-class labels by combining fraud status with device type
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Creating multi-class fraud labels...")
        
        # Create device-based fraud categories
        self.data['fraud_class'] = 'normal'
        
        # Map different device types for fraudulent transactions
        devices = self.data['device_used'].unique()
        for device in devices:
            mask = (self.data['is_fraud'] == True) & (self.data['device_used'] == device)
            self.data.loc[mask, 'fraud_class'] = f"{device}_fraud"
        
        # Create alternative multi-class labels based on payment channel
        self.data['fraud_payment_class'] = 'normal'
        channels = self.data['payment_channel'].unique()
        for channel in channels:
            mask = (self.data['is_fraud'] == True) & (self.data['payment_channel'] == channel)
            self.data.loc[mask, 'fraud_payment_class'] = f"{channel}_fraud"
        
        # Encode multi-class labels
        le_fraud_class = LabelEncoder()
        self.data['fraud_class_encoded'] = le_fraud_class.fit_transform(self.data['fraud_class'])
        
        self.class_mappings['fraud_class'] = {i: cls for i, cls in enumerate(le_fraud_class.classes_)}
        
        le_payment_class = LabelEncoder()
        self.data['fraud_payment_class_encoded'] = le_payment_class.fit_transform(self.data['fraud_payment_class'])
        
        self.class_mappings['fraud_payment_class'] = {i: cls for i, cls in enumerate(le_payment_class.classes_)}
        
        print("Fraud class mapping:")
        for code, name in self.class_mappings['fraud_class'].items():
            print(f"  {code} -> {name}")
        
        # Display class distribution
        print("Fraud class distribution:")
        print(self.data['fraud_class'].value_counts(normalize=True) * 100)
        
        return self.data
    
    def handle_imbalance(self, target_col='fraud_class_encoded', strategy='smote'):
        """
        Handle class imbalance using various strategies
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Handling class imbalance using {strategy}...")
        
        # Select features and target
        features = []
        
        # Add encoded categorical features
        for col in self.categorical_cols:
            if col + '_encoded' in self.data.columns:
                features.append(col + '_encoded')
        
        # Add numerical features
        for col in self.numerical_cols:
            if col in self.data.columns:
                features.append(col)
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:
            features.extend(['day_of_week', 'hour_of_day', 'month'])
        
        X = self.data[features]
        y = self.data[target_col]
        
        # Split data before applying SMOTE to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to training data only
        if strategy.lower() == 'smote':
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Original training class distribution: {pd.Series(y_train).value_counts()}")
            print(f"Resampled training class distribution: {pd.Series(y_train_resampled).value_counts()}")
            
            target_prefix = target_col.replace('_encoded', '')
            if target_prefix in self.class_mappings:
                print("Class distribution with names:")
                for class_code, count in pd.Series(y_train_resampled).value_counts().items():
                    class_name = self.class_mappings[target_prefix][class_code]
                    print(f"  {class_code} ({class_name}): {count}")
            
            return X_train_resampled, X_test, y_train_resampled, y_test, features
        
        # No resampling
        return X_train, X_test, y_train, y_test, features
    
    def prepare_for_binary_anomaly_detection(self):
        """
        Prepare data for binary anomaly detection
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preparing data for anomaly detection...")
        
        # Select relevant features
        features = []
        
        # Add encoded categorical features
        for col in self.categorical_cols:
            if col + '_encoded' in self.data.columns:
                features.append(col + '_encoded')
        
        # Add numerical features
        for col in self.numerical_cols:
            if col in self.data.columns:
                features.append(col)
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:
            features.extend(['day_of_week', 'hour_of_day', 'month'])
        
        X = self.data[features]
        y = self.data[self.label_col].astype(int)  # Binary target
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, features, scaler
    
    def prepare_for_time_series(self, target_feature='location'):
        """
        Prepare data for time series prediction
        Uses timestamp to create sequence-based features
        Target can be one of the metadata features (not IP or hash)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Preparing data for time series prediction of '{target_feature}'...")
        
        # Sort data by timestamp
        if 'timestamp' in self.data.columns:
            self.data = self.data.sort_values('timestamp')
        
        # Select features excluding IP address and device hash
        features = []
        
        # Add encoded categorical features except the target
        for col in self.categorical_cols:
            if col != target_feature and col + '_encoded' in self.data.columns:
                features.append(col + '_encoded')
        
        # Add numerical features
        for col in self.numerical_cols:
            if col in self.data.columns:
                features.append(col)
        
        # Add time-based features
        if 'day_of_week' in self.data.columns:
            features.extend(['day_of_week', 'hour_of_day', 'month'])
        
        X = self.data[features]
        
        # Prepare target
        if target_feature + '_encoded' in self.data.columns:
            y = self.data[target_feature + '_encoded']
        else:
            le = LabelEncoder()
            y = le.fit_transform(self.data[target_feature])
            # 保存编码映射关系
            self.class_mappings[target_feature] = {i: cls for i, cls in enumerate(le.classes_)}
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data while preserving time order
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, features, scaler
    
    def get_class_names(self, target_col='fraud_class_encoded'):
        """
        Get the sorted class names list in order of encoding
        """
        target_prefix = target_col.replace('_encoded', '')
        if target_prefix in self.class_mappings:
            mapping = self.class_mappings[target_prefix]
            # return the sorted class names list
            return [mapping[i] for i in range(len(mapping))]
        return None

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