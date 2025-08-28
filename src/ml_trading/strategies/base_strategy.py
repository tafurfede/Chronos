"""
Trading Strategy Base Classes
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.positions = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals from data"""
        pass
    
    @abstractmethod
    def calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence score for the signal"""
        pass

class MomentumStrategy(TradingStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, params: Dict = None):
        params = params or {
            'fast_ma': 20,
            'slow_ma': 50,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        super().__init__('Momentum', params)
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate momentum signals"""
        
        # Calculate indicators
        data['sma_fast'] = data['close'].rolling(self.params['fast_ma']).mean()
        data['sma_slow'] = data['close'].rolling(self.params['slow_ma']).mean()
        data['rsi'] = self._calculate_rsi(data['close'], self.params['rsi_period'])
        
        latest = data.iloc[-1]
        
        # Generate signal
        signal = 'hold'
        confidence = 0.5
        
        # Bullish conditions
        if (latest['sma_fast'] > latest['sma_slow'] and 
            latest['close'] > latest['sma_fast'] and
            latest['rsi'] < self.params['rsi_overbought']):
            signal = 'buy'
            confidence = self.calculate_confidence(data)
            
        # Bearish conditions
        elif (latest['sma_fast'] < latest['sma_slow'] and 
              latest['close'] < latest['sma_fast'] and
              latest['rsi'] > self.params['rsi_oversold']):
            signal = 'sell'
            confidence = self.calculate_confidence(data)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'indicators': {
                'sma_fast': latest['sma_fast'],
                'sma_slow': latest['sma_slow'],
                'rsi': latest['rsi']
            }
        }
    
    def calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence based on indicator alignment"""
        latest = data.iloc[-1]
        confidence = 0.5
        
        # Check trend strength
        if 'sma_fast' in data.columns:
            trend_strength = abs(latest['sma_fast'] - latest['sma_slow']) / latest['close']
            confidence += trend_strength * 2
        
        # Check RSI
        if 'rsi' in data.columns:
            if 30 < latest['rsi'] < 70:
                confidence += 0.1
        
        # Check volume
        if 'volume' in data.columns:
            vol_avg = data['volume'].rolling(20).mean().iloc[-1]
            if latest['volume'] > vol_avg * 1.5:
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, params: Dict = None):
        params = params or {
            'lookback': 20,
            'num_std': 2,
            'rsi_period': 14
        }
        super().__init__('MeanReversion', params)
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate mean reversion signals"""
        
        # Calculate Bollinger Bands
        data['sma'] = data['close'].rolling(self.params['lookback']).mean()
        data['std'] = data['close'].rolling(self.params['lookback']).std()
        data['upper_band'] = data['sma'] + (data['std'] * self.params['num_std'])
        data['lower_band'] = data['sma'] - (data['std'] * self.params['num_std'])
        
        latest = data.iloc[-1]
        
        signal = 'hold'
        confidence = 0.5
        
        # Oversold - potential buy
        if latest['close'] < latest['lower_band']:
            signal = 'buy'
            confidence = self.calculate_confidence(data)
            
        # Overbought - potential sell
        elif latest['close'] > latest['upper_band']:
            signal = 'sell'
            confidence = self.calculate_confidence(data)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'indicators': {
                'upper_band': latest['upper_band'],
                'lower_band': latest['lower_band'],
                'sma': latest['sma']
            }
        }
    
    def calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence for mean reversion"""
        latest = data.iloc[-1]
        
        # Distance from bands
        if latest['close'] < latest['lower_band']:
            distance = (latest['lower_band'] - latest['close']) / latest['close']
        elif latest['close'] > latest['upper_band']:
            distance = (latest['close'] - latest['upper_band']) / latest['close']
        else:
            distance = 0
        
        confidence = 0.5 + min(distance * 10, 0.4)
        return confidence

class EnsembleStrategy(TradingStrategy):
    """Combines multiple strategies"""
    
    def __init__(self, strategies: List[TradingStrategy]):
        super().__init__('Ensemble', {})
        self.strategies = strategies
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Combine signals from multiple strategies"""
        
        all_signals = []
        all_confidences = []
        
        for strategy in self.strategies:
            result = strategy.generate_signals(data)
            all_signals.append(result['signal'])
            all_confidences.append(result['confidence'])
        
        # Weighted voting
        buy_score = sum(c for s, c in zip(all_signals, all_confidences) if s == 'buy')
        sell_score = sum(c for s, c in zip(all_signals, all_confidences) if s == 'sell')
        
        if buy_score > sell_score and buy_score > 0.5:
            signal = 'buy'
            confidence = buy_score / len(self.strategies)
        elif sell_score > buy_score and sell_score > 0.5:
            signal = 'sell'
            confidence = sell_score / len(self.strategies)
        else:
            signal = 'hold'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strategy_votes': all_signals
        }
    
    def calculate_confidence(self, data: pd.DataFrame) -> float:
        """Average confidence from all strategies"""
        confidences = [s.calculate_confidence(data) for s in self.strategies]
        return np.mean(confidences)