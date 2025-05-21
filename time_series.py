import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
import os
os.makedirs('plots', exist_ok=True)

warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    def __init__(self):
        """
        Initialize time series predictor with LSTM and Random Forest models
        """
        self.models = {}
        self.results = {}
        self.scalers = {}
    
    def prepare_time_series_data(self, X, y, sequence_length=10):
        """
        Transform data into sequences for LSTM model
        """
        # First convert input data to numpy array to avoid Pandas index issues
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_values = X.values
        else:
            X_values = np.array(X)
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = np.array(y)
            
        X_seq, y_seq = [], []
        for i in range(len(X_values) - sequence_length):
            X_seq.append(X_values[i:i+sequence_length])
            y_seq.append(y_values[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, X_train, y_train, sequence_length=10, units=50, epochs=50, batch_size=32):
        """
        Train LSTM model for time series prediction
        """
        print("Training LSTM for time series prediction...")
        start_time = time.time()
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_time_series_data(X_train, y_train, sequence_length)
        
        print(f"Prepared {len(X_seq)} sequences of length {sequence_length}")
        print(f"Input shape: {X_seq.shape}, Target shape: {y_seq.shape}")
        
        # Create LSTM model
        model = Sequential([
            LSTM(units, activation='relu', return_sequences=True, 
                input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(units//2, activation='relu'),
            Dropout(0.2),
            Dense(1 if len(y_seq.shape) == 1 else y_seq.shape[1])
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store model and training history
        self.models['lstm'] = model
        self.results['lstm_history'] = history.history
        
        train_time = time.time() - start_time
        print(f"LSTM training completed in {train_time:.2f} seconds")
        
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        """
        Train Random Forest for time series prediction
        """
        print("Training Random Forest for time series prediction...")
        start_time = time.time()
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Store model
        self.models['random_forest'] = model
        
        # Get feature importance
        importance = dict(zip(
            [f"f{i}" for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        self.results['rf_importance'] = importance
        
        train_time = time.time() - start_time
        print(f"Random Forest training completed in {train_time:.2f} seconds")
        
        return model
    
    def predict_lstm(self, X_test, y_test, sequence_length=10):
        """
        Make predictions using trained LSTM model
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM model not found. Train the model first.")
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_time_series_data(X_test, y_test, sequence_length)
        
        # Get predictions
        model = self.models['lstm']
        y_pred = model.predict(X_seq)
        
        # Flatten predictions if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        
        # Store results
        self.results['lstm_true'] = y_seq
        self.results['lstm_pred'] = y_pred
        
        return y_pred, y_seq
    
    def predict_random_forest(self, X_test, y_test):
        """
        Make predictions using trained Random Forest model
        """
        if 'random_forest' not in self.models:
            raise ValueError("Random Forest model not found. Train the model first.")
        
        # Get predictions
        model = self.models['random_forest']
        y_pred = model.predict(X_test)
        
        # Store results
        self.results['rf_true'] = y_test
        self.results['rf_pred'] = y_pred
        
        return y_pred
    
    def evaluate_lstm(self, y_true, y_pred, is_classification=False):
        """
        Evaluate LSTM model performance
        """
        print("Evaluating LSTM model...")
        
        if is_classification:
            # For classification tasks
            # Round predictions to nearest integer for comparison
            y_pred_classes = np.round(y_pred).astype(int)
            accuracy = np.mean(y_pred_classes == y_true)
            print(f"Accuracy: {accuracy:.4f}")
            
            results = {
                'accuracy': accuracy,
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred)
            }
        else:
            # For regression tasks
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        self.results['lstm_metrics'] = results
        return results
    
    def evaluate_random_forest(self, y_true, y_pred, is_classification=False):
        """
        Evaluate Random Forest model performance
        """
        print("Evaluating Random Forest model...")
        
        if is_classification:
            # For classification tasks
            # Round predictions to nearest integer for comparison
            y_pred_classes = np.round(y_pred).astype(int)
            accuracy = np.mean(y_pred_classes == y_true)
            print(f"Accuracy: {accuracy:.4f}")
            
            results = {
                'accuracy': accuracy,
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred)
            }
        else:
            # For regression tasks
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        self.results['rf_metrics'] = results
        return results
    
    def plot_predictions(self, model_name, num_samples=100, is_classification=False):
        """
        Plot actual vs predicted values
        """
        if model_name == 'lstm':
            y_true = self.results.get('lstm_true')
            y_pred = self.results.get('lstm_pred')
            title = 'LSTM'
        elif model_name == 'random_forest':
            y_true = self.results.get('rf_true')
            y_pred = self.results.get('rf_pred')
            title = 'Random Forest'
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if y_true is None or y_pred is None:
            raise ValueError(f"No predictions found for {model_name}. Run prediction first.")
        
        # Plot a sample of the predictions
        plt.figure(figsize=(12, 6))
        
        # If classification, make a scatter plot
        if is_classification:
            plt.scatter(range(num_samples), y_true[:num_samples], label='Actual', alpha=0.7)
            plt.scatter(range(num_samples), y_pred[:num_samples], label='Predicted', alpha=0.7)
            plt.ylabel('Class')
            plt.ylim(-0.5, max(np.max(y_true), np.max(y_pred)) + 0.5)
        else:
            # If regression, plot lines
            plt.plot(range(num_samples), y_true[:num_samples], label='Actual', marker='o')
            plt.plot(range(num_samples), y_pred[:num_samples], label='Predicted', marker='x')
            plt.ylabel('Value')
        
        plt.title(f'{title} - Actual vs Predicted Values')
        plt.xlabel('Sample Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'plots/{model_name}_predictions.png'
        plt.savefig(filename)
        plt.close()
        print(f"Predictions plot saved to {filename}")
    
    def plot_training_history(self):
        """
        Plot LSTM training history
        """
        if 'lstm_history' not in self.results:
            raise ValueError("No training history found. Train LSTM model first.")
        
        history = self.results['lstm_history']
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('LSTM Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'plots/lstm_training_history.png'
        plt.savefig(filename)
        plt.close()  
        print(f"Training history plot saved to {filename}")
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance from Random Forest model
        """
        if 'rf_importance' not in self.results:
            raise ValueError("No feature importance found. Train Random Forest model first.")
        
        importance = self.results['rf_importance']
        
        if feature_names is not None:
            # Map feature indices to names
            importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Take top 20 features
        features = list(importance.keys())[:20]
        values = list(importance.values())[:20]
        
        plt.figure(figsize=(12, 8))
        plt.barh(features, values)
        plt.title('Random Forest - Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        
        filename = 'plots/rf_feature_importance.png'
        plt.savefig(filename)
        plt.close()  
        print(f"Feature importance plot saved to {filename}")
    
    def compare_models(self, metric='rmse'):
        """
        Compare models based on specified metric
        """
        if 'lstm_metrics' not in self.results or 'rf_metrics' not in self.results:
            raise ValueError("Both models must be evaluated before comparison.")
        
        lstm_metrics = self.results['lstm_metrics']
        rf_metrics = self.results['rf_metrics']
        
        # Check if metric exists in both results
        if metric not in lstm_metrics or metric not in rf_metrics:
            raise ValueError(f"Metric {metric} not found in results.")
        
        # Compare
        metrics = {
            'LSTM': lstm_metrics[metric],
            'Random Forest': rf_metrics[metric]
        }
        
        # For metrics where lower is better (like MSE, RMSE, MAE)
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae']
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values())
        
        # Color bars based on better/worse
        if lower_is_better:
            best_model = min(metrics, key=metrics.get)
            for i, (model, _) in enumerate(metrics.items()):
                bars[i].set_color('green' if model == best_model else 'blue')
        else:  # Higher is better (like R²)
            best_model = max(metrics, key=metrics.get)
            for i, (model, _) in enumerate(metrics.items()):
                bars[i].set_color('green' if model == best_model else 'blue')
        
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):
            plt.text(i, value + (0.01 * (1 if value >= 0 else -1)), f'{value:.4f}', ha='center')
        
        plt.tight_layout()
        filename = f'plots/model_comparison_{metric}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Model comparison plot saved to {filename}")
        
        return metrics

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