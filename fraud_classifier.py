import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import time
import warnings
import os
os.makedirs('plots', exist_ok=True)

warnings.filterwarnings('ignore')

class FraudClassifier:
    def __init__(self):
        """
        Initialize fraud classifier with XGBoost and Random Forest models
        """
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        
    def train_xgboost(self, X_train, y_train, params=None):
        """
        Train XGBoost for multiclass fraud classification
        """
        print("Training XGBoost classifier...")
        start_time = time.time()
        
        # Use default parameters if none provided
        if params is None:
            params = {
                'objective': 'multi:softprob',  # Multiclass probability
                'num_class': len(np.unique(y_train)),  # Number of classes
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'eval_metric': 'mlogloss',
                'seed': 1
            } 
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train the model
        num_rounds = 100
        model = xgb.train(params, dtrain, num_rounds)
        
        # Store the model
        self.models['xgboost'] = model
        
        train_time = time.time() - start_time
        print(f"XGBoost training completed in {train_time:.2f} seconds")
        
        # Get feature importance
        importance = model.get_score(importance_type='weight')
        self.feature_importances['xgboost'] = importance
        
        return model
    
    def train_random_forest(self, X_train, y_train, params=None):
        """
        Train Random Forest for multiclass fraud classification
        """
        print("Training Random Forest classifier...")
        start_time = time.time()
        
        # Use default parameters if none provided
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42
            }
        
        # Create and train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['random_forest'] = model
        
        train_time = time.time() - start_time
        print(f"Random Forest training completed in {train_time:.2f} seconds")
        
        # Get feature importance
        importance = dict(zip(
            [f"f{i}" for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        self.feature_importances['random_forest'] = importance
        
        return model
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost'):
        """
        Perform hyperparameter optimization using GridSearchCV
        """
        if model_type.lower() == 'xgboost':
            print("Optimizing XGBoost hyperparameters...")
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.1, 0.01],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'n_estimators': [100, 200]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(np.unique(y_train)),
                random_state=42
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='f1_weighted',
                cv=3,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_params_
            
        elif model_type.lower() == 'random_forest':
            print("Optimizing Random Forest hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf_model = RandomForestClassifier(random_state=42)
            
            grid_search = GridSearchCV(
                estimator=rf_model,
                param_grid=param_grid,
                scoring='f1_weighted',
                cv=3,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_params_
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def evaluate_model(self, model_name, X_test, y_test, feature_names=None, class_names=None):
        """
        Evaluate model performance on test data
        """
        print(f"Evaluating {model_name} model...")
        
        # Get the model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        # Make predictions
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_prob = model.predict(dtest)
            y_pred = np.argmax(y_prob, axis=1)
        else:  # Random Forest
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # calculate the recall of each class
        n_classes = len(np.unique(y_test))
        recall_per_class = []
        precision_per_class = []
        f1_per_class = []
        
        for i in range(n_classes):
            # the true positives, false negatives, and false positives of the current class
            true_positives = np.sum((y_test == i) & (y_pred == i))
            false_negatives = np.sum((y_test == i) & (y_pred != i))
            false_positives = np.sum((y_test != i) & (y_pred == i))
            
            # calculate the recall, precision, and F1 of the current class
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            recall_per_class.append(recall)
            precision_per_class.append(precision)
            f1_per_class.append(f1)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'f1_per_class': f1_per_class
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision_macro:.4f}")
        print(f"Recall (macro): {recall_macro:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        
        # print the recall of each class
        if class_names is not None:
            print("\nRecall of each class:")
            for i, class_name in enumerate(class_names):
                print(f"{class_name}: {recall_per_class[i]:.4f}")
        
        # Classification report
        if class_names is not None:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        return results
    
    def plot_confusion_matrix(self, model_name, class_names=None):
        """
        Plot confusion matrix for the specified model
        """
        results = self.results.get(model_name)
        if results is None:
            raise ValueError(f"No evaluation results found for {model_name}. Evaluate the model first.")
        
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names if class_names is not None else "auto",
                    yticklabels=class_names if class_names is not None else "auto")
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{model_name}_confusion_matrix.png'))
    
    def plot_feature_importance(self, model_name, feature_names=None):
        """
        Plot feature importance for the specified model
        """
        importance = self.feature_importances.get(model_name)
        if importance is None:
            raise ValueError(f"No feature importance found for {model_name}.")
        
        # Modify feature name mapping logic
        if feature_names is not None and model_name == 'xgboost':
            # If XGBoost feature names are a mix of full feature names and f+numbers
            feature_importance = {}
            for k, v in importance.items():
                # Try to determine if it's f+number format
                if k.startswith('f') and k[1:].isdigit():
                    # If it's f+number format, use index mapping
                    feature_idx = int(k[1:])
                    if feature_idx < len(feature_names):
                        feature_importance[feature_names[feature_idx]] = v
                else:
                    # If it's already the actual feature name, use it directly
                    feature_importance[k] = v
            importance = feature_importance
        elif feature_names is not None and model_name == 'random_forest':
            # Random Forest feature importance processing remains unchanged
            importance = dict(zip(
                feature_names,
                [importance[f"f{i}"] for i in range(len(feature_names)) if f"f{i}" in importance]
            ))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Plot
        plt.figure(figsize=(12, 8))
        # Ensure there is content to display
        if len(importance) > 0:
            # Limit to top 20 features
            keys_to_plot = list(importance.keys())[:min(20, len(importance))]
            values_to_plot = [importance[k] for k in keys_to_plot]
            plt.barh(keys_to_plot, values_to_plot)
            plt.title(f'Top {len(keys_to_plot)} Feature Importance - {model_name}')
        else:
            plt.title(f'No Feature Importance Available - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{model_name}_feature_importance.png'))
    
    def plot_roc_curves(self, model_name, class_names=None):
        """
        Plot ROC curves for multiclass classification
        """
        results = self.results.get(model_name)
        if results is None:
            raise ValueError(f"No evaluation results found for {model_name}. Evaluate the model first.")
        
        y_test = results['y_true']
        y_prob = results['y_prob']
        
        n_classes = y_prob.shape[1]
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            # Get class probabilities and true values in one-vs-rest fashion
            y_true_bin = (y_test == i).astype(int)
            y_score = y_prob[:, i]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            class_label = class_names[i] if class_names is not None else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, 
                     label=f'ROC curve - {class_label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{model_name}_roc_curves.png'))
    
    def plot_recall_by_class(self, class_names=None):
        """
        Plot the recall of each model on each class
        """
        if len(self.results) < 1:
            raise ValueError("No model evaluation results. Please evaluate the model first.")
        
        # collect the recall of each model on each class
        recall_data = {}
        for model_name, results in self.results.items():
            if 'recall_per_class' in results:
                recall_data[model_name] = results['recall_per_class']
        
        if not recall_data:
            raise ValueError("No recall data found for any model")
        
        # set the class names
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(next(iter(recall_data.values()))))]
        
        plt.figure(figsize=(14, 8))
        
        # set the position and width of the bar chart
        n_models = len(recall_data)
        bar_width = 0.8 / n_models
        index = np.arange(len(class_names))
        
        # plot the recall of each model
        for i, (model_name, recalls) in enumerate(recall_data.items()):
            position = index + i * bar_width
            plt.bar(position, recalls, bar_width, label=model_name)
            
            # add the value on the bar
            for j, recall in enumerate(recalls):
                plt.text(position[j], recall + 0.02, f'{recall:.2f}', 
                         ha='center', va='bottom', fontsize=9, rotation=0)
        
        # set the chart properties
        plt.xlabel('Class')
        plt.ylabel('Recall')
        plt.title('Recall of each model on each class')
        plt.xticks(index + bar_width * (n_models-1) / 2, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)  # Recall range is between 0 and 1
        plt.tight_layout()
        
        # save the chart
        plt.savefig(os.path.join('plots', 'recall_by_class_comparison.png'))
        plt.close()
        print("Recall of each model on each class comparison chart saved")
    
    def compare_models(self, metric='f1_weighted'):
        """
        Compare models based on the specified metric
        """
        if not self.results:
            raise ValueError("No evaluation results found. Evaluate models first.")
        
        metrics = {}
        for model_name, results in self.results.items():
            metrics[model_name] = results[metric]
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'model_comparison_{metric}.png'))
        
        return metrics

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