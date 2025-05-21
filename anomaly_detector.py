import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
import os
os.makedirs('plots', exist_ok=True)

warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        """
        Initialize anomaly detector with autoencoder and KMeans models
        """
        self.models = {}
        self.results = {}
        self.thresholds = {}
    
    def train_autoencoder(self, X_train, y_train=None, hidden_dims=[64, 32, 16], 
                          activation='relu', epochs=50, batch_size=256):
        """
        Train deep learning autoencoder for anomaly detection
        """
        print("Training Autoencoder for anomaly detection...")
        start_time = time.time()
        
        # Get input dimensions
        input_dim = X_train.shape[1]
        
        # Create encoder layers
        input_layer = Input(shape=(input_dim,))
        
        # Build encoder
        encoder = input_layer
        for dim in hidden_dims:
            encoder = Dense(dim, activation=activation)(encoder)
            encoder = Dropout(0.2)(encoder)
        
        # Build decoder (symmetrical to encoder)
        decoder = encoder
        for dim in reversed(hidden_dims[:-1]):
            decoder = Dense(dim, activation=activation)(decoder)
            decoder = Dropout(0.2)(decoder)
        
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoder)
        
        # Create model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        # Print model summary
        print(autoencoder.summary())
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store the model
        self.models['autoencoder'] = autoencoder
        
        # Calculate reconstruction error
        reconstructions = autoencoder.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        
        # If true labels are provided, calculate threshold using precision-recall trade-off
        if y_train is not None:
            # Find a good threshold based on F1 score
            thresholds = np.linspace(np.min(mse), np.max(mse), 100)
            best_f1 = 0
            best_threshold = 0
            
            for threshold in thresholds:
                y_pred = (mse > threshold).astype(int)
                if np.sum(y_pred) > 0 and np.sum(y_train) > 0:
                    f1 = f1_score(y_train, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            print(f"Best threshold: {best_threshold:.6f} with F1: {best_f1:.4f}")
            self.thresholds['autoencoder'] = best_threshold
        else:
            # Use simple statistical approach: mean + 2*std
            # threshold = np.mean(mse) + 2 * np.std(mse)
            threshold = np.mean(mse) + np.std(mse)
            self.thresholds['autoencoder'] = threshold
            print(f"Statistical threshold: {threshold:.6f}")
        
        train_time = time.time() - start_time
        print(f"Autoencoder training completed in {train_time:.2f} seconds")
        
        # Store training history
        self.results['autoencoder_history'] = history.history
        
        return autoencoder
    
    def train_kmeans(self, X_train, y_train=None, n_clusters=2):
        """
        Train KMeans for anomaly detection
        """
        print("Training KMeans for anomaly detection...")
        start_time = time.time()
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_train)
        
        # Store the model
        self.models['kmeans'] = kmeans
        
        # Calculate cluster centers and distances
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:
            silhouette = silhouette_score(X_train, cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
        
        # Calculate distance to cluster center for each point
        distances = np.zeros(X_train.shape[0])
        for i in range(X_train.shape[0]):
            cluster_idx = cluster_labels[i]
            distances[i] = np.linalg.norm(X_train[i] - cluster_centers[cluster_idx])
        
        # If true labels are provided, determine which cluster is the fraud cluster
        if y_train is not None:
            # Check each cluster for fraud ratio
            fraud_ratios = {}
            for cluster in range(n_clusters):
                cluster_samples = (cluster_labels == cluster)
                if np.sum(cluster_samples) > 0:
                    fraud_ratio = np.mean(y_train[cluster_samples])
                    fraud_ratios[cluster] = fraud_ratio
                    print(f"Cluster {cluster}: {np.sum(cluster_samples)} samples, fraud ratio: {fraud_ratio:.4f}")
            
            # Identify the fraud cluster (highest fraud ratio)
            fraud_cluster = max(fraud_ratios, key=fraud_ratios.get)
            print(f"Identified fraud cluster: {fraud_cluster}")
            
            # Store fraud cluster
            self.results['fraud_cluster'] = fraud_cluster
            
            # Define distance threshold based on this cluster
            cluster_points = X_train[cluster_labels == fraud_cluster]
            if len(cluster_points) > 0:
                threshold = np.percentile(
                    np.linalg.norm(cluster_points - cluster_centers[fraud_cluster], axis=1), 
                    75
                )
                self.thresholds['kmeans'] = threshold
                print(f"Distance threshold for fraud cluster: {threshold:.6f}")
        else:
            # Determine fraud cluster based on distances (assume fraud points are further from their centers)
            mean_distances = []
            for cluster in range(n_clusters):
                cluster_samples = (cluster_labels == cluster)
                if np.sum(cluster_samples) > 0:
                    mean_dist = np.mean(distances[cluster_samples])
                    mean_distances.append((cluster, mean_dist))
            
            # Cluster with highest average distance is likely the fraud cluster
            fraud_cluster = max(mean_distances, key=lambda x: x[1])[0]
            print(f"Assumed fraud cluster (based on distances): {fraud_cluster}")
            
            # Store fraud cluster
            self.results['fraud_cluster'] = fraud_cluster
            
            # Define distance threshold
            threshold = np.percentile(distances, 70)  # Assume top 10% are anomalies
            self.thresholds['kmeans'] = threshold
            print(f"Distance threshold (statistical): {threshold:.6f}")
        
        train_time = time.time() - start_time
        print(f"KMeans training completed in {train_time:.2f} seconds")
        
        return kmeans
    
    def detect_anomalies_autoencoder(self, X_test, threshold=None):
        """
        Detect anomalies using the trained autoencoder
        """
        if 'autoencoder' not in self.models:
            raise ValueError("Autoencoder model not found. Train the model first.")
        
        # Use stored threshold if not provided
        if threshold is None:
            threshold = self.thresholds.get('autoencoder')
            if threshold is None:
                raise ValueError("No threshold found for autoencoder. Train with labels or provide threshold.")
        
        # Get model
        autoencoder = self.models['autoencoder']
        
        # Calculate reconstruction error
        reconstructions = autoencoder.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        # Classify as anomaly if error > threshold
        anomalies = (mse > threshold).astype(int)
        
        # Store reconstruction errors
        self.results['autoencoder_errors'] = mse
        
        return anomalies, mse
    
    def detect_anomalies_kmeans(self, X_test, threshold=None):
        """
        Detect anomalies using the trained KMeans model
        """
        if 'kmeans' not in self.models:
            raise ValueError("KMeans model not found. Train the model first.")
        
        # Use stored threshold if not provided
        if threshold is None:
            threshold = self.thresholds.get('kmeans')
            if threshold is None:
                raise ValueError("No threshold found for KMeans. Train with labels or provide threshold.")
        
        # Get model and fraud cluster
        kmeans = self.models['kmeans']
        fraud_cluster = self.results.get('fraud_cluster')
        
        if fraud_cluster is None:
            raise ValueError("Fraud cluster not identified. Train with labels first.")
        
        # Predict clusters
        cluster_labels = kmeans.predict(X_test)
        
        # Calculate distances to assigned cluster centers
        distances = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            cluster_idx = cluster_labels[i]
            distances[i] = np.linalg.norm(X_test[i] - kmeans.cluster_centers_[cluster_idx])
        
        # Method 1: Points in fraud cluster with distance > threshold
        anomalies_1 = ((cluster_labels == fraud_cluster) & (distances > threshold)).astype(int)
        
        # Method 2: Points in fraud cluster
        anomalies_2 = (cluster_labels == fraud_cluster).astype(int)
        
        # Store the distances
        self.results['kmeans_distances'] = distances
        self.results['kmeans_clusters'] = cluster_labels
        
        # Return the better method based on previous validation
        return anomalies_2, distances
    
    def evaluate_detector(self, method, y_pred, y_true, scores=None):
        """
        Evaluate anomaly detection performance
        """
        print(f"Evaluating {method} anomaly detection...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Store results
        self.results[f'{method}_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        if scores is not None:
            self.results[f'{method}_scores'] = scores
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return self.results[f'{method}_metrics']
    
    def plot_reconstruction_error(self, method='autoencoder', y_true=None):
        """
        Plot reconstruction error distribution for autoencoder
        """
        if method == 'autoencoder':
            errors = self.results.get('autoencoder_errors')
            if errors is None:
                raise ValueError("No reconstruction errors found. Run detect_anomalies_autoencoder first.")
            
            threshold = self.thresholds.get('autoencoder')
            title = 'Autoencoder Reconstruction Error'
            xlabel = 'Reconstruction Error (MSE)'
        
        elif method == 'kmeans':
            errors = self.results.get('kmeans_distances')
            if errors is None:
                raise ValueError("No distances found. Run detect_anomalies_kmeans first.")
            
            threshold = self.thresholds.get('kmeans')
            title = 'KMeans Distance to Cluster Center'
            xlabel = 'Distance'
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        plt.figure(figsize=(12, 6))
        
        if y_true is not None:
            # Plot distribution by class
            plt.hist(errors[y_true == 0], bins=50, alpha=0.5, color='blue', label='Normal')
            plt.hist(errors[y_true == 1], bins=50, alpha=0.5, color='red', label='Fraud')
            plt.legend()
        else:
            # Plot overall distribution
            plt.hist(errors, bins=50, alpha=0.7)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')
            plt.legend()
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{method}_reconstruction_error.png'))
    
    def plot_clusters(self, X, method='kmeans', y_true=None):
        """
        Plot clusters in 2D space (using PCA reduction)
        """
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        
        if method == 'kmeans':
            if 'kmeans_clusters' not in self.results:
                raise ValueError("No cluster assignments found. Run detect_anomalies_kmeans first.")
            
            cluster_labels = self.results['kmeans_clusters']
            
            # Plot each cluster
            for cluster in np.unique(cluster_labels):
                cluster_points = X_2d[cluster_labels == cluster]
                plt.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1], 
                    label=f'Cluster {cluster}', 
                    alpha=0.6
                )
            
            # Plot cluster centers
            kmeans = self.models['kmeans']
            centers_2d = pca.transform(kmeans.cluster_centers_)
            plt.scatter(
                centers_2d[:, 0], 
                centers_2d[:, 1], 
                s=200, 
                marker='X', 
                c='black', 
                label='Centroids'
            )
            
            title = 'KMeans Clusters (PCA 2D projection)'
        
        elif method == 'autoencoder':
            if 'autoencoder_errors' not in self.results:
                raise ValueError("No reconstruction errors found. Run detect_anomalies_autoencoder first.")
            
            errors = self.results['autoencoder_errors']
            threshold = self.thresholds.get('autoencoder')
            
            # Color by reconstruction error
            sc = plt.scatter(
                X_2d[:, 0], 
                X_2d[:, 1], 
                c=errors, 
                cmap='viridis', 
                alpha=0.6
            )
            plt.colorbar(sc, label='Reconstruction Error')
            
            if threshold is not None:
                # Mark points above threshold
                anomalies = errors > threshold
                plt.scatter(
                    X_2d[anomalies, 0], 
                    X_2d[anomalies, 1], 
                    s=50, 
                    edgecolors='red', 
                    facecolors='none', 
                    linewidths=2, 
                    label='Detected Anomalies'
                )
            
            title = 'Autoencoder Anomalies (PCA 2D projection)'
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # If true labels provided, add contour
        if y_true is not None:
            plt.scatter(
                X_2d[y_true == 1, 0], 
                X_2d[y_true == 1, 1], 
                s=50, 
                edgecolors='red', 
                facecolors='none', 
                linewidths=2, 
                label='True Fraud'
            )
        
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{method}_clusters.png'))
    
    def compare_methods(self, metric='f1'):
        """
        Compare both anomaly detection methods
        """
        if f'autoencoder_metrics' not in self.results or f'kmeans_metrics' not in self.results:
            raise ValueError("Both methods must be evaluated before comparison.")
        
        # Get metrics
        autoencoder_metrics = self.results['autoencoder_metrics']
        kmeans_metrics = self.results['kmeans_metrics']
        
        # Compare
        metrics = {
            'Autoencoder': autoencoder_metrics[metric],
            'KMeans': kmeans_metrics[metric]
        }
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title(f'Anomaly Detection Methods Comparison - {metric}')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        
        # Add values on top of bars
        for i, (model, value) in enumerate(metrics.items()):
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'comparison.png'))
        
        return metrics

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