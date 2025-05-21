import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# Import custom modules
from data_processor import DataProcessor
from fraud_classifier import FraudClassifier
from anomaly_detector import AnomalyDetector
from time_series import TimeSeriesPredictor

# Suppress warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self, data_path=None, sample_size=None):
        """
        Initialize the fraud detection system
        """
        self.processor = DataProcessor(data_path)
        self.sample_size = sample_size
        
        # Placeholders for other components
        self.classifier = None
        self.anomaly_detector = None
        self.time_series_predictor = None
        
        # Results container
        self.results = {}
    
    def load_and_preprocess_data(self, create_multiclass=True):
        """
        Load and preprocess the dataset
        """
        print("\n=== 数据加载和预处理 ===")
        self.processor.load_data(sample_size=self.sample_size)
        self.processor.preprocess_data()
        
        if create_multiclass:
            self.processor.create_multiclass_labels()
    
    def run_fraud_classification(self, optimize=False):
        """
        Execute multi-class fraud classification task
        """
        print("\n=== 任务1: 多类欺诈检测 ===")
        
        # Prepare data for classification
        X_train, X_test, y_train, y_test, features = self.processor.handle_imbalance(
            target_col='fraud_class_encoded'
        )
        
        # use the correct class mapping to get the class names
        class_names = self.processor.get_class_names('fraud_class_encoded')
        if class_names:
            print(f"List of class names: {class_names}")
        else:
            # if cannot get the mapping, use the original class names
            class_names = self.processor.data['fraud_class'].unique()
            print(f"Class information (not mapped): {class_names}")
        
        # Initialize classifier
        self.classifier = FraudClassifier()
        
        # Optimize hyperparameters if requested
        if optimize:
            print("\n优化XGBoost超参数...")
            xgb_params = self.classifier.optimize_hyperparameters(
                X_train, y_train, model_type='xgboost'
            )
            
            print("\n优化随机森林超参数...")
            rf_params = self.classifier.optimize_hyperparameters(
                X_train, y_train, model_type='random_forest'
            )
        else:
            xgb_params = None
            rf_params = None
        
        # Train XGBoost
        print("\n训练XGBoost模型...")
        xgb_model = self.classifier.train_xgboost(X_train, y_train, params=xgb_params)
        
        # Train Random Forest
        print("\n训练随机森林模型...")
        rf_model = self.classifier.train_random_forest(X_train, y_train, params=rf_params)
        
        # Evaluate models
        print("\n评估XGBoost模型...")
        xgb_results = self.classifier.evaluate_model(
            'xgboost', X_test, y_test, 
            feature_names=features, class_names=class_names
        )
        
        print("\n评估随机森林模型...")
        rf_results = self.classifier.evaluate_model(
            'random_forest', X_test, y_test, 
            feature_names=features, class_names=class_names
        )
        
        # Compare models
        print("\n比较两种分类模型...")
        comparison = self.classifier.compare_models(metric='f1_weighted')
        
        # Recall comparison
        print("\n比较模型召回率...")
        recall_comparison = self.classifier.compare_models(metric='recall_macro')
        
        # Plot confusion matrices
        print("\n绘制混淆矩阵...")
        self.classifier.plot_confusion_matrix('xgboost', class_names=class_names)
        self.classifier.plot_confusion_matrix('random_forest', class_names=class_names)
        
        # Plot feature importance
        print("\n绘制特征重要性...")
        self.classifier.plot_feature_importance('xgboost', feature_names=features)
        self.classifier.plot_feature_importance('random_forest', feature_names=features)
        
        # Plot ROC curves
        print("\n绘制ROC曲线...")
        self.classifier.plot_roc_curves('xgboost', class_names=class_names)
        self.classifier.plot_roc_curves('random_forest', class_names=class_names)
        
        # Plot recall by class
        print("\n绘制各类别召回率...")
        self.classifier.plot_recall_by_class(class_names=class_names)
        
        # Store results
        self.results['classification'] = {
            'xgboost': xgb_results,
            'random_forest': rf_results,
            'comparison': comparison,
            'recall_comparison': recall_comparison,
            'class_names': class_names
        }
        
        return self.results['classification']
    
    def run_anomaly_detection(self):
        """
        Execute fraud anomaly detection task
        """
        print("\n=== 任务2: 欺诈异常检测 ===")
        
        # Prepare data for anomaly detection
        X_train, X_test, y_train, y_test, features, scaler = self.processor.prepare_for_binary_anomaly_detection()
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Train autoencoder
        print("\n训练自编码器...")
        autoencoder = self.anomaly_detector.train_autoencoder(
            X_train, y_train, hidden_dims=[64, 32, 16], epochs=20
        )
        
        # Train KMeans
        print("\n训练K-means聚类...")
        kmeans = self.anomaly_detector.train_kmeans(X_train, y_train, n_clusters=2)
        
        # Detect anomalies
        print("\n使用自编码器检测异常...")
        anomalies_ae, scores_ae = self.anomaly_detector.detect_anomalies_autoencoder(X_test)
        
        print("\n使用K-means检测异常...")
        anomalies_km, scores_km = self.anomaly_detector.detect_anomalies_kmeans(X_test)
        
        # Evaluate performance
        print("\n评估自编码器性能...")
        ae_results = self.anomaly_detector.evaluate_detector('autoencoder', anomalies_ae, y_test, scores_ae)
        
        print("\n评估K-means性能...")
        km_results = self.anomaly_detector.evaluate_detector('kmeans', anomalies_km, y_test, scores_km)
        
        # Compare methods
        print("\n比较两种异常检测方法...")
        comparison = self.anomaly_detector.compare_methods(metric='f1')
        
        # 添加召回率对比
        print("\n比较异常检测方法召回率...")
        recall_comparison = self.anomaly_detector.compare_methods(metric='recall')
        
        # Plot error distributions
        print("\n绘制重建误差分布...")
        self.anomaly_detector.plot_reconstruction_error('autoencoder', y_test)
        
        # Plot clusters
        print("\n绘制聚类结果...")
        self.anomaly_detector.plot_clusters(X_test, 'kmeans', y_test)
        
        # Store results
        self.results['anomaly_detection'] = {
            'autoencoder': ae_results,
            'kmeans': km_results,
            'comparison': comparison,
            'recall_comparison': recall_comparison
        }
        
        return self.results['anomaly_detection']
    
    def run_time_series_prediction(self, target_feature='location'):
        """
        Execute time series prediction task
        """
        print(f"\n=== 任务3: 时间序列预测 ({target_feature}) ===")
        
        # Prepare data for time series prediction
        X_train, X_test, y_train, y_test, features, scaler = self.processor.prepare_for_time_series(
            target_feature=target_feature
        )
        
        # Check if doing classification or regression
        is_classification = len(np.unique(y_train)) < 10  # Assume classification if few unique values
        print(f"任务类型: {'分类' if is_classification else '回归'}")
        
        # 获取目标特征的类别映射(如果是分类任务)
        target_classes = None
        if is_classification and target_feature in self.processor.class_mappings:
            target_classes = self.processor.get_class_names(target_feature + '_encoded')
            print(f"目标类别: {target_classes}")
        
        # Initialize time series predictor
        self.time_series_predictor = TimeSeriesPredictor()
        
        # Train Random Forest (it's faster)
        print("\n训练随机森林...")
        rf_model = self.time_series_predictor.train_random_forest(X_train, y_train)
        rf_preds = self.time_series_predictor.predict_random_forest(X_test, y_test)
        
        # Train LSTM with smaller sequence length for demonstration
        print("\n训练LSTM...")
        sequence_length = 5
        lstm_model = self.time_series_predictor.train_lstm(
            X_train, y_train, sequence_length=sequence_length, epochs=20
        )
        lstm_preds, lstm_true = self.time_series_predictor.predict_lstm(
            X_test, y_test, sequence_length=sequence_length
        )
        
        # Evaluate models
        print("\n评估随机森林性能...")
        rf_results = self.time_series_predictor.evaluate_random_forest(
            y_test, rf_preds, is_classification
        )
        
        print("\n评估LSTM性能...")
        lstm_results = self.time_series_predictor.evaluate_lstm(
            lstm_true, lstm_preds, is_classification
        )
        
        # Compare models
        print("\n比较两种时间序列预测模型...")
        comparison = self.time_series_predictor.compare_models(
            metric='rmse' if not is_classification else 'mae'
        )
        
        # Plot predictions
        print("\n绘制预测结果...")
        self.time_series_predictor.plot_predictions(
            'random_forest', is_classification=is_classification
        )
        self.time_series_predictor.plot_predictions(
            'lstm', is_classification=is_classification
        )
        
        # Plot feature importance
        print("\n绘制特征重要性...")
        self.time_series_predictor.plot_feature_importance(feature_names=features)
        
        # Plot training history for LSTM
        print("\n绘制LSTM训练历史...")
        self.time_series_predictor.plot_training_history()
        
        # Store results
        self.results['time_series'] = {
            'target_feature': target_feature,
            'random_forest': rf_results,
            'lstm': lstm_results,
            'comparison': comparison,
            'is_classification': is_classification,
            'target_classes': target_classes
        }
        
        return self.results['time_series']
    
    def run_all_tasks(self, optimize=False):
        """
        Run all three fraud detection tasks
        """
        print("\n=== 启动完整欺诈检测系统 ===")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run task 1: Multi-class fraud classification
        self.run_fraud_classification(optimize=optimize)
        
        # Run task 2: Anomaly detection
        self.run_anomaly_detection()
        
        # Run task 3: Time series prediction
        self.run_time_series_prediction(target_feature='location')
        
        print("\n=== 全部任务完成 ===")
        return self.results
    
    def print_summary(self):
        """
        Print a summary of all results
        """
        print("\n\n=== 欺诈检测系统结果总结 ===")
        
        if 'classification' in self.results:
            print("\n-- 多类欺诈检测结果 --")
            xgb_f1 = self.results['classification']['xgboost']['f1_weighted']
            rf_f1 = self.results['classification']['random_forest']['f1_weighted']
            xgb_recall = self.results['classification']['xgboost']['recall_macro']
            rf_recall = self.results['classification']['random_forest']['recall_macro']
            
            print(f"XGBoost F1加权分数: {xgb_f1:.4f}")
            print(f"随机森林 F1加权分数: {rf_f1:.4f}")
            print(f"XGBoost 召回率(宏平均): {xgb_recall:.4f}")
            print(f"随机森林 召回率(宏平均): {rf_recall:.4f}")
            print(f"最佳模型(F1): {'XGBoost' if xgb_f1 > rf_f1 else '随机森林'}")
            print(f"最佳模型(召回率): {'XGBoost' if xgb_recall > rf_recall else '随机森林'}")
        
        if 'anomaly_detection' in self.results:
            print("\n-- 欺诈异常检测结果 --")
            ae_f1 = self.results['anomaly_detection']['autoencoder']['f1']
            km_f1 = self.results['anomaly_detection']['kmeans']['f1']
            ae_recall = self.results['anomaly_detection']['autoencoder']['recall']
            km_recall = self.results['anomaly_detection']['kmeans']['recall']
            
            print(f"自编码器 F1分数: {ae_f1:.4f}")
            print(f"K-means F1分数: {km_f1:.4f}")
            print(f"自编码器 召回率: {ae_recall:.4f}")
            print(f"K-means 召回率: {km_recall:.4f}")
            print(f"最佳模型(F1): {'自编码器' if ae_f1 > km_f1 else 'K-means'}")
            print(f"最佳模型(召回率): {'自编码器' if ae_recall > km_recall else 'K-means'}")
        
        if 'time_series' in self.results:
            print("\n-- 时间序列预测结果 --")
            target = self.results['time_series']['target_feature']
            is_classification = self.results['time_series']['is_classification']
            
            if is_classification:
                rf_metric = self.results['time_series']['random_forest']['accuracy']
                lstm_metric = self.results['time_series']['lstm']['accuracy']
                metric_name = "准确率"
            else:
                rf_metric = self.results['time_series']['random_forest']['rmse']
                lstm_metric = self.results['time_series']['lstm']['rmse']
                metric_name = "RMSE"
            
            print(f"预测目标: {target}")
            print(f"任务类型: {'分类' if is_classification else '回归'}")
            print(f"随机森林 {metric_name}: {rf_metric:.4f}")
            print(f"LSTM {metric_name}: {lstm_metric:.4f}")
            
            if is_classification:
                print(f"最佳模型: {'LSTM' if lstm_metric > rf_metric else '随机森林'}")
            else:
                print(f"最佳模型: {'LSTM' if lstm_metric < rf_metric else '随机森林'}")


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='金融欺诈检测系统')
    
    parser.add_argument('--data_path', type=str, 
                        default='data/financial_fraud_detection_dataset.csv',
                        help='数据集路径')
    
    parser.add_argument('--sample_size', type=int, default=None,
                        help='样本大小 (设置为 None 使用全部数据)')
    
    parser.add_argument('--task', type=str, choices=['all', 'classification', 'anomaly', 'time_series'],
                        default='all', help='要运行的任务')
    
    parser.add_argument('--optimize', action='store_true',
                        help='是否进行超参数优化')
    
    parser.add_argument('--target_feature', type=str, default='location',
                        help='时间序列预测的目标特征')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize system
    system = FraudDetectionSystem(data_path=args.data_path, sample_size=args.sample_size)
    
    # Load and preprocess data
    system.load_and_preprocess_data()
    
    # Run requested task(s)
    if args.task == 'all':
        system.run_all_tasks(optimize=args.optimize)
    elif args.task == 'classification':
        system.run_fraud_classification(optimize=args.optimize)
    elif args.task == 'anomaly':
        system.run_anomaly_detection()
    elif args.task == 'time_series':
        system.run_time_series_prediction(target_feature=args.target_feature)
    
    # Print summary
    system.print_summary()