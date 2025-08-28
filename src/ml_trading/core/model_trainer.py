#!/usr/bin/env python3
"""
Model Training Pipeline for ML Trading System
Handles training, validation, and optimization of all ML models
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import json
from pathlib import Path

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Deep Learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Local imports
from .ml_trading_system import MLFeatureEngine, MLEnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training pipeline
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_engine = MLFeatureEngine()
        self.ensemble = MLEnsembleModel()
        self.training_history = {}
        
    def load_training_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """Load and prepare training data"""
        
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Loading data for {symbol}")
            
            # Load from CSV or fetch from API
            file_path = self.data_dir / f"raw/{symbol}_data.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            else:
                df = self._fetch_historical_data(symbol)
                df.to_csv(file_path, index=False)
            
            # Generate features
            features = self.feature_engine.generate_features(df, symbol)
            
            # Add target variable (next day return)
            features['target'] = df['close'].pct_change().shift(-1)
            
            # Add symbol identifier
            features['symbol'] = symbol
            
            all_data.append(features)
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove NaN values
        combined_data = combined_data.dropna()
        
        logger.info(f"Loaded {len(combined_data)} samples for training")
        
        return combined_data
    
    def _fetch_historical_data(self, symbol: str, days: int = 730) -> pd.DataFrame:
        """Fetch historical data from API"""
        
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")
        
        # Rename columns to match expected format
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'date': 'timestamp'})
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def prepare_train_test_split(self, data: pd.DataFrame, 
                                test_size: float = 0.2,
                                val_size: float = 0.1) -> Tuple:
        """Prepare train/validation/test splits for time series"""
        
        # Sort by timestamp
        data = data.sort_values('timestamp') if 'timestamp' in data.columns else data
        
        # Calculate split indices
        n = len(data)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        # Split data
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        # Prepare features and targets
        feature_cols = [col for col in data.columns 
                       if col not in ['target', 'symbol', 'timestamp']]
        
        X_train = train_data[feature_cols].values
        y_train = train_data['target'].values
        
        X_val = val_data[feature_cols].values
        y_val = val_data['target'].values
        
        X_test = test_data[feature_cols].values
        y_test = test_data['target'].values
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
        """Train XGBoost model with hyperparameter optimization"""
        
        logger.info("Training XGBoost model...")
        
        # Hyperparameter search space
        param_dist = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Base model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random search
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        # Fit model
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Train final model with early stopping (XGBoost 2.0+ syntax)
        best_model.set_params(early_stopping_rounds=50)
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        logger.info(f"XGBoost - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
        logger.info(f"Best params: {random_search.best_params_}")
        
        return best_model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        
        logger.info("Training LightGBM model...")
        
        # Hyperparameter search space
        param_dist = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_samples': [5, 10, 20]
        }
        
        # Base model
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random search
        random_search = RandomizedSearchCV(
            lgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        # Fit model
        random_search.fit(X_train, y_train)
        
        # Get best model and retrain with early stopping
        best_model = random_search.best_estimator_
        
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        logger.info(f"LightGBM - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
        
        return best_model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, n_features: int):
        """Train deep neural network"""
        
        logger.info("Training Neural Network...")
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_dim=n_features),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ModelCheckpoint(
                str(self.models_dir / 'best_nn_model.h5'),
                save_best_only=True
            ),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        
        logger.info(f"Neural Network - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return model, history
    
    def train_lstm(self, X_train, y_train, X_val, y_val, n_features: int):
        """Train LSTM model for time series"""
        
        logger.info("Training LSTM model...")
        
        # Reshape data for LSTM (samples, timesteps, features)
        sequence_length = 60
        
        X_train_seq = self._create_sequences(X_train, sequence_length)
        X_val_seq = self._create_sequences(X_val, sequence_length)
        
        # Adjust targets
        y_train_seq = y_train[sequence_length:]
        y_val_seq = y_val[sequence_length:]
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, 
                                input_shape=(sequence_length, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ModelCheckpoint(
                str(self.models_dir / 'best_lstm_model.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info("LSTM training complete")
        
        return model, history
    
    def _create_sequences(self, X, sequence_length):
        """Create sequences for LSTM"""
        sequences = []
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
        return np.array(sequences)
    
    def train_all_models(self, data: pd.DataFrame) -> Dict:
        """Train all models in the ensemble"""
        
        logger.info("Starting complete model training pipeline...")
        
        # Prepare data
        (X_train, y_train, X_val, y_val, 
         X_test, y_test, feature_cols) = self.prepare_train_test_split(data)
        
        n_features = len(feature_cols)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, self.models_dir / 'scaler.pkl')
        
        # Train models
        models = {}
        
        # XGBoost
        models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # LightGBM
        models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Neural Network
        nn_model, nn_history = self.train_neural_network(
            X_train, y_train, X_val, y_val, n_features
        )
        models['neural_network'] = nn_model
        
        # LSTM
        lstm_model, lstm_history = self.train_lstm(
            X_train, y_train, X_val, y_val, n_features
        )
        models['lstm'] = lstm_model
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluate_ensemble(
            models, X_test, y_test, feature_cols
        )
        
        # Save training results
        self.save_training_results(models, ensemble_metrics, feature_cols)
        
        return {
            'models': models,
            'metrics': ensemble_metrics,
            'feature_columns': feature_cols
        }
    
    def evaluate_ensemble(self, models: Dict, X_test, y_test, 
                         feature_cols: List[str]) -> Dict:
        """Evaluate ensemble performance"""
        
        logger.info("Evaluating ensemble performance...")
        
        predictions = {}
        metrics = {}
        
        # Get predictions from each model
        for name, model in models.items():
            if name in ['lstm']:
                # LSTM needs sequences
                X_test_seq = self._create_sequences(X_test, 60)
                y_test_seq = y_test[60:]
                pred = model.predict(X_test_seq).flatten()
                mse = mean_squared_error(y_test_seq, pred)
            else:
                pred = model.predict(X_test)
                mse = mean_squared_error(y_test, pred)
            
            predictions[name] = pred
            metrics[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_test[:len(pred)], pred)
            }
            
            logger.info(f"{name} - MSE: {mse:.6f}, RMSE: {np.sqrt(mse):.6f}")
        
        # Calculate ensemble prediction (weighted average)
        # For simplicity, using equal weights
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        ensemble_mse = mean_squared_error(y_test[:len(ensemble_pred)], ensemble_pred)
        
        metrics['ensemble'] = {
            'mse': ensemble_mse,
            'rmse': np.sqrt(ensemble_mse),
            'mae': mean_absolute_error(y_test[:len(ensemble_pred)], ensemble_pred)
        }
        
        logger.info(f"Ensemble - MSE: {ensemble_mse:.6f}, RMSE: {np.sqrt(ensemble_mse):.6f}")
        
        # Calculate trading metrics
        trading_metrics = self.calculate_trading_metrics(ensemble_pred, y_test[:len(ensemble_pred)])
        metrics['trading'] = trading_metrics
        
        return metrics
    
    def calculate_trading_metrics(self, predictions, actual) -> Dict:
        """Calculate trading-specific metrics"""
        
        # Convert to binary signals
        pred_signals = (predictions > 0).astype(int)
        actual_signals = (actual > 0).astype(int)
        
        # Accuracy
        accuracy = np.mean(pred_signals == actual_signals)
        
        # Precision for long signals
        true_positives = np.sum((pred_signals == 1) & (actual_signals == 1))
        false_positives = np.sum((pred_signals == 1) & (actual_signals == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Calculate returns
        strategy_returns = predictions * actual
        
        # Sharpe ratio (assuming daily returns)
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
        }
    
    def save_models(self, models: Dict):
        """Save all trained models"""
        
        logger.info("Saving models...")
        
        for name, model in models.items():
            if name in ['neural_network', 'lstm']:
                # Keras models
                model.save(self.models_dir / f'{name}_model.h5')
            else:
                # Sklearn-compatible models
                joblib.dump(model, self.models_dir / f'{name}_model.pkl')
        
        logger.info(f"Models saved to {self.models_dir}")
    
    def save_training_results(self, models: Dict, metrics: Dict, 
                             feature_cols: List[str]):
        """Save training results and metadata"""
        
        # Save metrics
        with open(self.models_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save feature columns
        with open(self.models_dir / 'feature_columns.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        # Save models
        self.save_models(models)
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_cols),
            'models': list(models.keys()),
            'best_model': min(metrics.items(), key=lambda x: x[1].get('mse', float('inf')))[0]
        }
        
        with open(self.models_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Training results saved successfully")
    
    def load_models(self) -> Dict:
        """Load saved models"""
        
        models = {}
        
        # Load XGBoost
        xgb_path = self.models_dir / 'xgboost_model.pkl'
        if xgb_path.exists():
            models['xgboost'] = joblib.load(xgb_path)
        
        # Load LightGBM
        lgb_path = self.models_dir / 'lightgbm_model.pkl'
        if lgb_path.exists():
            models['lightgbm'] = joblib.load(lgb_path)
        
        # Load Neural Network
        nn_path = self.models_dir / 'neural_network_model.h5'
        if nn_path.exists():
            models['neural_network'] = tf.keras.models.load_model(nn_path)
        
        # Load LSTM
        lstm_path = self.models_dir / 'lstm_model.h5'
        if lstm_path.exists():
            models['lstm'] = tf.keras.models.load_model(lstm_path)
        
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        
        logger.info(f"Loaded {len(models)} models")
        
        return models, scaler


if __name__ == "__main__":
    # Test training pipeline
    trainer = ModelTrainer()
    
    # Load data
    data = trainer.load_training_data()
    
    # Train models
    results = trainer.train_all_models(data)
    
    print("\nTraining Complete!")
    print(f"Models trained: {list(results['models'].keys())}")
    print(f"Ensemble MSE: {results['metrics']['ensemble']['mse']:.6f}")
    print(f"Trading Accuracy: {results['metrics']['trading']['accuracy']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['trading']['sharpe']:.2f}")