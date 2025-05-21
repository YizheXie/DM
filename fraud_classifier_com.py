import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库
import matplotlib  # 导入绘图基础库
matplotlib.use('Agg')  # 设置matplotlib后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt  # 导入matplotlib的绘图功能
import seaborn as sns  # 导入高级绘图库
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 导入评估指标
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score  # 导入更多评估指标
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # 导入ROC和PR曲线相关函数
import xgboost as xgb  # 导入XGBoost库
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证
import time  # 导入时间库用于计时
import warnings  # 导入警告控制
import os  # 导入操作系统接口
os.makedirs('plots', exist_ok=True)  # 创建plots文件夹，如果已存在则不报错

warnings.filterwarnings('ignore')  # 忽略所有警告信息

class FraudClassifier:
    def __init__(self):
        """
        Initialize fraud classifier with XGBoost and Random Forest models
        """
        self.models = {}  # 初始化存储模型的字典
        self.results = {}  # 初始化存储结果的字典
        self.feature_importances = {}  # 初始化存储特征重要性的字典
        
    def train_xgboost(self, X_train, y_train, params=None):
        """
        Train XGBoost for multiclass fraud classification
        """
        print("Training XGBoost classifier...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Use default parameters if none provided
        if params is None:  # 如果没有提供参数
            params = {  # 设置默认参数
                'objective': 'multi:softprob',  # 多分类概率输出
                'num_class': len(np.unique(y_train)),  # 类别数量
                'n_estimators': 100,  # 树的数量
                'learning_rate': 0.1,  # 学习率
                'max_depth': 6,  # 树的最大深度
                'min_child_weight': 1,  # 最小子节点权重
                'subsample': 0.8,  # 样本采样比例
                'colsample_bytree': 0.8,  # 特征采样比例
                'gamma': 0,  # 节点分裂的最小损失减少
                'reg_alpha': 0,  # L1正则化项
                'reg_lambda': 1,  # L2正则化项
                'eval_metric': 'mlogloss',  # 评估指标
                'seed': 1  # 随机种子
            } 
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)  # 创建XGBoost的DMatrix数据结构
        
        # Train the model
        num_rounds = 100  # 设置训练轮数
        model = xgb.train(params, dtrain, num_rounds)  # 训练XGBoost模型
        
        # Store the model
        self.models['xgboost'] = model  # 存储训练好的XGBoost模型
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"XGBoost training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        # Get feature importance
        importance = model.get_score(importance_type='weight')  # 获取特征重要性
        self.feature_importances['xgboost'] = importance  # 存储XGBoost的特征重要性
        
        return model  # 返回训练好的模型
        
        # 函数功能总结：训练XGBoost模型用于多分类欺诈检测，可使用自定义参数或默认参数，并存储模型及特征重要性信息。
    
    def train_random_forest(self, X_train, y_train, params=None):
        """
        Train Random Forest for multiclass fraud classification
        """
        print("Training Random Forest classifier...")  # 打印训练开始的信息
        start_time = time.time()  # 记录训练开始时间
        
        # Use default parameters if none provided
        if params is None:  # 如果没有提供参数
            params = {  # 设置默认参数
                'n_estimators': 100,  # 树的数量
                'max_depth': 10,  # 树的最大深度
                'min_samples_split': 2,  # 内部节点再划分所需的最小样本数
                'min_samples_leaf': 1,  # 叶节点所需的最小样本数
                'max_features': 'sqrt',  # 寻找最佳分割点时考虑的特征数（平方根）
                'bootstrap': True,  # 是否使用bootstrap样本
                'class_weight': 'balanced',  # 类别权重设置为平衡模式
                'random_state': 42  # 随机种子
            }
        
        # Create and train the model
        model = RandomForestClassifier(**params)  # 创建随机森林分类器
        model.fit(X_train, y_train)  # 训练随机森林模型
        
        # Store the model
        self.models['random_forest'] = model  # 存储训练好的随机森林模型
        
        train_time = time.time() - start_time  # 计算训练耗时
        print(f"Random Forest training completed in {train_time:.2f} seconds")  # 打印训练完成信息
        
        # Get feature importance
        importance = dict(zip(  # 创建特征重要性字典
            [f"f{i}" for i in range(X_train.shape[1])],  # 特征名称（f0, f1, ...）
            model.feature_importances_  # 特征重要性值
        ))
        self.feature_importances['random_forest'] = importance  # 存储随机森林的特征重要性
        
        return model  # 返回训练好的模型
        
        # 函数功能总结：训练随机森林模型用于多分类欺诈检测，可使用自定义参数或默认参数，并存储模型及特征重要性信息。
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost'):
        """
        Perform hyperparameter optimization using GridSearchCV
        """
        if model_type.lower() == 'xgboost':  # 如果是XGBoost模型
            print("Optimizing XGBoost hyperparameters...")  # 打印超参数优化的信息
            param_grid = {  # 设置参数网格
                'max_depth': [3, 6, 9],  # 树的最大深度选项
                'learning_rate': [0.1, 0.01],  # 学习率选项
                'subsample': [0.8, 1.0],  # 样本采样比例选项
                'colsample_bytree': [0.8, 1.0],  # 特征采样比例选项
                'n_estimators': [100, 200]  # 树的数量选项
            }
            
            xgb_model = xgb.XGBClassifier(  # 创建XGBoost分类器
                objective='multi:softprob',  # 多分类概率输出
                num_class=len(np.unique(y_train)),  # 类别数量
                random_state=42  # 随机种子
            )
            
            grid_search = GridSearchCV(  # 创建网格搜索对象
                estimator=xgb_model,  # 使用XGBoost模型
                param_grid=param_grid,  # 参数网格
                scoring='f1_weighted',  # 使用加权F1分数作为评估指标
                cv=3,  # 3折交叉验证
                verbose=1  # 显示进度
            )
            
            grid_search.fit(X_train, y_train)  # 执行网格搜索
            print(f"Best parameters: {grid_search.best_params_}")  # 打印最佳参数
            return grid_search.best_params_  # 返回最佳参数
            
        elif model_type.lower() == 'random_forest':  # 如果是随机森林模型
            print("Optimizing Random Forest hyperparameters...")  # 打印超参数优化的信息
            param_grid = {  # 设置参数网格
                'n_estimators': [100, 200],  # 树的数量选项
                'max_depth': [10, 20, None],  # 树的最大深度选项
                'min_samples_split': [2, 5],  # 内部节点再划分所需的最小样本数选项
                'min_samples_leaf': [1, 2]  # 叶节点所需的最小样本数选项
            }
            
            rf_model = RandomForestClassifier(random_state=42)  # 创建随机森林分类器
            
            grid_search = GridSearchCV(  # 创建网格搜索对象
                estimator=rf_model,  # 使用随机森林模型
                param_grid=param_grid,  # 参数网格
                scoring='f1_weighted',  # 使用加权F1分数作为评估指标
                cv=3,  # 3折交叉验证
                verbose=1  # 显示进度
            )
            
            grid_search.fit(X_train, y_train)  # 执行网格搜索
            print(f"Best parameters: {grid_search.best_params_}")  # 打印最佳参数
            return grid_search.best_params_  # 返回最佳参数
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")  # 如果模型类型不支持则抛出错误
            
        # 函数功能总结：使用网格搜索交叉验证对XGBoost或随机森林模型进行超参数优化，并返回最佳参数集。
    
    def evaluate_model(self, model_name, X_test, y_test, feature_names=None, class_names=None):
        """
        Evaluate model performance on test data
        """
        print(f"Evaluating {model_name} model...")  # 打印评估开始的信息
        
        # Get the model
        model = self.models.get(model_name)  # 获取指定名称的模型
        if model is None:  # 如果模型不存在
            raise ValueError(f"Model {model_name} not found. Train the model first.")  # 抛出错误
        
        # Make predictions
        if model_name == 'xgboost':  # 如果是XGBoost模型
            dtest = xgb.DMatrix(X_test)  # 创建XGBoost的DMatrix测试数据
            y_prob = model.predict(dtest)  # 预测概率
            y_pred = np.argmax(y_prob, axis=1)  # 获取最大概率的类别索引作为预测结果
        else:  # 如果是随机森林模型
            y_pred = model.predict(X_test)  # 预测类别
            y_prob = model.predict_proba(X_test)  # 预测概率
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
        precision_macro = precision_score(y_test, y_pred, average='macro')  # 计算宏平均精确率
        recall_macro = recall_score(y_test, y_pred, average='macro')  # 计算宏平均召回率
        f1_macro = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted')  # 计算加权平均精确率
        recall_weighted = recall_score(y_test, y_pred, average='weighted')  # 计算加权平均召回率
        f1_weighted = f1_score(y_test, y_pred, average='weighted')  # 计算加权平均F1分数
        
        # calculate the recall of each class
        n_classes = len(np.unique(y_test))  # 获取类别数量
        recall_per_class = []  # 初始化每个类别的召回率列表
        precision_per_class = []  # 初始化每个类别的精确率列表
        f1_per_class = []  # 初始化每个类别的F1分数列表
        
        for i in range(n_classes):  # 遍历每个类别
            # the true positives, false negatives, and false positives of the current class
            true_positives = np.sum((y_test == i) & (y_pred == i))  # 计算真正例数量
            false_negatives = np.sum((y_test == i) & (y_pred != i))  # 计算假负例数量
            false_positives = np.sum((y_test != i) & (y_pred == i))  # 计算假正例数量
            
            # calculate the recall, precision, and F1 of the current class
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0  # 计算召回率
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0  # 计算精确率
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # 计算F1分数
            
            recall_per_class.append(recall)  # 添加到召回率列表
            precision_per_class.append(precision)  # 添加到精确率列表
            f1_per_class.append(f1)  # 添加到F1分数列表
        
        # Store results
        results = {  # 创建结果字典
            'accuracy': accuracy,  # 存储准确率
            'precision_macro': precision_macro,  # 存储宏平均精确率
            'recall_macro': recall_macro,  # 存储宏平均召回率
            'f1_macro': f1_macro,  # 存储宏平均F1分数
            'precision_weighted': precision_weighted,  # 存储加权平均精确率
            'recall_weighted': recall_weighted,  # 存储加权平均召回率
            'f1_weighted': f1_weighted,  # 存储加权平均F1分数
            'confusion_matrix': confusion_matrix(y_test, y_pred),  # 存储混淆矩阵
            'y_true': y_test,  # 存储真实标签
            'y_pred': y_pred,  # 存储预测标签
            'y_prob': y_prob,  # 存储预测概率
            'recall_per_class': recall_per_class,  # 存储每个类别的召回率
            'precision_per_class': precision_per_class,  # 存储每个类别的精确率
            'f1_per_class': f1_per_class  # 存储每个类别的F1分数
        }
        
        self.results[model_name] = results  # 将结果存储到模型名称对应的结果字典中
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")  # 打印准确率
        print(f"Precision (macro): {precision_macro:.4f}")  # 打印宏平均精确率
        print(f"Recall (macro): {recall_macro:.4f}")  # 打印宏平均召回率
        print(f"F1 (macro): {f1_macro:.4f}")  # 打印宏平均F1分数
        print(f"Precision (weighted): {precision_weighted:.4f}")  # 打印加权平均精确率
        print(f"Recall (weighted): {recall_weighted:.4f}")  # 打印加权平均召回率
        print(f"F1 (weighted): {f1_weighted:.4f}")  # 打印加权平均F1分数
        
        # print the recall of each class
        if class_names is not None:  # 如果提供了类别名称
            print("\nRecall of each class:")  # 打印每个类别的召回率标题
            for i, class_name in enumerate(class_names):  # 遍历每个类别名称和索引
                print(f"{class_name}: {recall_per_class[i]:.4f}")  # 打印类别名称和召回率
        
        # Classification report
        if class_names is not None:  # 如果提供了类别名称
            print("\nClassification Report:")  # 打印分类报告标题
            print(classification_report(y_test, y_pred, target_names=class_names))  # 打印带有类别名称的分类报告
        else:
            print("\nClassification Report:")  # 打印分类报告标题
            print(classification_report(y_test, y_pred))  # 打印分类报告
        
        return results  # 返回评估结果
        
        # 函数功能总结：评估模型在测试集上的性能，计算各种评估指标（准确率、精确率、召回率、F1分数等），并打印详细的评估结果。
    
    def plot_confusion_matrix(self, model_name, class_names=None):
        """
        Plot confusion matrix for the specified model
        """
        results = self.results.get(model_name)  # 获取指定模型的结果
        if results is None:  # 如果结果不存在
            raise ValueError(f"No evaluation results found for {model_name}. Evaluate the model first.")  # 抛出错误
        
        cm = results['confusion_matrix']  # 获取混淆矩阵
        
        plt.figure(figsize=(10, 8))  # 创建图表，设置大小
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',   # 绘制热图
                    xticklabels=class_names if class_names is not None else "auto",  # 设置x轴标签
                    yticklabels=class_names if class_names is not None else "auto")  # 设置y轴标签
        plt.title(f'Confusion Matrix - {model_name}')  # 设置标题
        plt.ylabel('True Label')  # 设置y轴标题
        plt.xlabel('Predicted Label')  # 设置x轴标题
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'{model_name}_confusion_matrix.png'))  # 保存图表
        
        # 函数功能总结：绘制指定模型的混淆矩阵热图，并保存为PNG文件。
    
    def plot_feature_importance(self, model_name, feature_names=None):
        """
        Plot feature importance for the specified model
        """
        importance = self.feature_importances.get(model_name)  # 获取指定模型的特征重要性
        if importance is None:  # 如果特征重要性不存在
            raise ValueError(f"No feature importance found for {model_name}.")  # 抛出错误
        
        # Modify feature name mapping logic
        if feature_names is not None and model_name == 'xgboost':  # 如果提供了特征名称且是XGBoost模型
            # If XGBoost feature names are a mix of full feature names and f+numbers
            feature_importance = {}  # 初始化特征重要性字典
            for k, v in importance.items():  # 遍历原始特征重要性字典
                # Try to determine if it's f+number format
                if k.startswith('f') and k[1:].isdigit():  # 如果键以'f'开头且后面是数字
                    # If it's f+number format, use index mapping
                    feature_idx = int(k[1:])  # 获取特征索引
                    if feature_idx < len(feature_names):  # 如果索引在特征名称范围内
                        feature_importance[feature_names[feature_idx]] = v  # 使用特征名称作为键
                else:
                    # If it's already the actual feature name, use it directly
                    feature_importance[k] = v  # 直接使用原始键
            importance = feature_importance  # 更新特征重要性字典
        elif feature_names is not None and model_name == 'random_forest':  # 如果提供了特征名称且是随机森林模型
            # Random Forest feature importance processing remains unchanged
            importance = dict(zip(  # 创建新的特征重要性字典
                feature_names,  # 使用特征名称
                [importance[f"f{i}"] for i in range(len(feature_names)) if f"f{i}" in importance]  # 获取对应索引的特征重要性值
            ))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))  # 按重要性降序排序
        
        # Plot
        plt.figure(figsize=(12, 8))  # 创建图表，设置大小
        # Ensure there is content to display
        if len(importance) > 0:  # 如果有特征重要性数据
            # Limit to top 20 features
            keys_to_plot = list(importance.keys())[:min(20, len(importance))]  # 获取前20个特征名称
            values_to_plot = [importance[k] for k in keys_to_plot]  # 获取对应的特征重要性值
            plt.barh(keys_to_plot, values_to_plot)  # 绘制水平条形图
            plt.title(f'Top {len(keys_to_plot)} Feature Importance - {model_name}')  # 设置标题
        else:
            plt.title(f'No Feature Importance Available - {model_name}')  # 如果没有数据，设置无数据标题
        plt.xlabel('Importance')  # 设置x轴标题
        plt.ylabel('Feature')  # 设置y轴标题
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'{model_name}_feature_importance.png'))  # 保存图表
        
        # 函数功能总结：绘制指定模型的特征重要性条形图，显示最重要的特征（最多20个），并保存为PNG文件。
    
    def plot_roc_curves(self, model_name, class_names=None):
        """
        Plot ROC curves for multiclass classification
        """
        results = self.results.get(model_name)  # 获取指定模型的结果
        if results is None:  # 如果结果不存在
            raise ValueError(f"No evaluation results found for {model_name}. Evaluate the model first.")  # 抛出错误
        
        y_test = results['y_true']  # 获取真实标签
        y_prob = results['y_prob']  # 获取预测概率
        
        n_classes = y_prob.shape[1]  # 获取类别数量
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))  # 创建图表，设置大小
        
        for i in range(n_classes):  # 遍历每个类别
            # Get class probabilities and true values in one-vs-rest fashion
            y_true_bin = (y_test == i).astype(int)  # 将真实标签转换为二分类（当前类别为1，其他为0）
            y_score = y_prob[:, i]  # 获取当前类别的预测概率
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)  # 计算ROC曲线的假正例率和真正例率
            roc_auc = auc(fpr, tpr)  # 计算AUC值
            
            # Plot
            class_label = class_names[i] if class_names is not None else f"Class {i}"  # 获取类别标签
            plt.plot(fpr, tpr, lw=2,   # 绘制ROC曲线
                     label=f'ROC curve - {class_label} (AUC = {roc_auc:.2f})')  # 设置标签，包含AUC值
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制对角线（随机猜测的基准）
        plt.xlim([0.0, 1.0])  # 设置x轴范围
        plt.ylim([0.0, 1.05])  # 设置y轴范围
        plt.xlabel('False Positive Rate')  # 设置x轴标题
        plt.ylabel('True Positive Rate')  # 设置y轴标题
        plt.title(f'ROC Curves - {model_name}')  # 设置标题
        plt.legend(loc="lower right")  # 添加图例，位置在右下角
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'{model_name}_roc_curves.png'))  # 保存图表
        
        # 函数功能总结：为多分类问题绘制ROC曲线，显示每个类别的ROC曲线和对应的AUC值，并保存为PNG文件。
    
    def plot_recall_by_class(self, class_names=None):
        """
        Plot the recall of each model on each class
        """
        if len(self.results) < 1:  # 如果没有模型结果
            raise ValueError("No model evaluation results. Please evaluate the model first.")  # 抛出错误
        
        # collect the recall of each model on each class
        recall_data = {}  # 初始化存储每个模型在每个类别上的召回率的字典
        for model_name, results in self.results.items():  # 遍历每个模型的结果
            if 'recall_per_class' in results:  # 如果结果中包含每个类别的召回率
                recall_data[model_name] = results['recall_per_class']  # 存储该模型的召回率
        
        if not recall_data:  # 如果没有找到召回率数据
            raise ValueError("No recall data found for any model")  # 抛出错误
        
        # set the class names
        if class_names is None:  # 如果没有提供类别名称
            class_names = [f"Class {i}" for i in range(len(next(iter(recall_data.values()))))]  # 创建默认类别名称
        
        plt.figure(figsize=(14, 8))  # 创建图表，设置大小
        
        # set the position and width of the bar chart
        n_models = len(recall_data)  # 获取模型数量
        bar_width = 0.8 / n_models  # 计算条形图宽度
        index = np.arange(len(class_names))  # 创建类别索引数组
        
        # plot the recall of each model
        for i, (model_name, recalls) in enumerate(recall_data.items()):  # 遍历每个模型及其召回率
            position = index + i * bar_width  # 计算条形图位置
            plt.bar(position, recalls, bar_width, label=model_name)  # 绘制条形图
            
            # add the value on the bar
            for j, recall in enumerate(recalls):  # 遍历每个类别的召回率
                plt.text(position[j], recall + 0.02, f'{recall:.2f}',   # 在条形图上添加文本
                         ha='center', va='bottom', fontsize=9, rotation=0)  # 设置文本样式
        
        # set the chart properties
        plt.xlabel('Class')  # 设置x轴标题
        plt.ylabel('Recall')  # 设置y轴标题
        plt.title('Recall of each model on each class')  # 设置标题
        plt.xticks(index + bar_width * (n_models-1) / 2, class_names, rotation=45, ha='right')  # 设置x轴刻度
        plt.legend()  # 添加图例
        plt.grid(axis='y', alpha=0.3)  # 添加水平网格线
        plt.ylim(0, 1.1)  # 设置y轴范围（召回率范围为0到1）
        plt.tight_layout()  # 优化图表布局
        
        # save the chart
        plt.savefig(os.path.join('plots', 'recall_by_class_comparison.png'))  # 保存图表
        plt.close()  # 关闭图表
        print("Recall of each model on each class comparison chart saved")  # 打印保存信息
        
        # 函数功能总结：绘制每个模型在每个类别上的召回率对比条形图，直观显示不同模型在不同类别上的表现，并保存为PNG文件。
    
    def compare_models(self, metric='f1_weighted'):
        """
        Compare models based on the specified metric
        """
        if not self.results:  # 如果没有模型结果
            raise ValueError("No evaluation results found. Evaluate models first.")  # 抛出错误
        
        metrics = {}  # 初始化存储每个模型指定指标的字典
        for model_name, results in self.results.items():  # 遍历每个模型的结果
            metrics[model_name] = results[metric]  # 存储指定指标的值
        
        # Plot comparison
        plt.figure(figsize=(10, 6))  # 创建图表，设置大小
        plt.bar(metrics.keys(), metrics.values())  # 绘制条形图
        plt.title(f'Model Comparison - {metric}')  # 设置标题
        plt.ylabel(metric)  # 设置y轴标题
        plt.ylim(0, 1)  # 设置y轴范围（大多数指标范围为0到1）
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):  # 遍历每个模型及其指标值
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center')  # 在条形图上方添加文本
        
        plt.tight_layout()  # 优化图表布局
        plt.savefig(os.path.join('plots', f'model_comparison_{metric}.png'))  # 保存图表
        
        return metrics  # 返回指标字典
        
        # 函数功能总结：根据指定的评估指标比较不同模型的性能，绘制比较条形图，并保存为PNG文件。

# # Example usage
# if __name__ == "__main__":
#     from data_processor import DataProcessor
    
#     # Process data
#     processor = DataProcessor()
#     data = processor.load_data(sample_size=100000)
#     data = processor.preprocess_data()
#     data = processor.create_multiclass_labels()
    
#     # Prepare data for classification
#     X_train, X_test, y_train, y_test, features = processor.handle_imbalance(target_col='fraud_class_encoded')
    
#     # Get class names
#     class_names = processor.get_class_names('fraud_class_encoded')
#     if class_names is None:
#         class_names = processor.data['fraud_class'].unique()
    
#     print(f"Class names in order: {class_names}")
    
#     # Train and evaluate models
#     classifier = FraudClassifier()
    
#     # Train XGBoost
#     xgb_model = classifier.train_xgboost(X_train, y_train)
#     xgb_results = classifier.evaluate_model('xgboost', X_test, y_test, feature_names=features, class_names=class_names)
    
#     # Train Random Forest
#     rf_model = classifier.train_random_forest(X_train, y_train)
#     rf_results = classifier.evaluate_model('random_forest', X_test, y_test, feature_names=features, class_names=class_names)
    
#     # Compare models
#     comparison = classifier.compare_models(metric='f1_weighted')
    
#     # Plot confusion matrices
#     classifier.plot_confusion_matrix('xgboost', class_names=class_names)
#     classifier.plot_confusion_matrix('random_forest', class_names=class_names)
    
#     # Plot recall by class
#     classifier.plot_recall_by_class(class_names=class_names) 