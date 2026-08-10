import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Tuple, List, Optional
import bisect


def parse_iso8601_to_seconds(iso_string: str) -> int:
    """Parse ISO 8601 timestamp to Unix seconds."""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return int(dt.timestamp())


class BtcPriceRecord:
    def __init__(self, timestamp: int, close: float):
        self.timestamp = timestamp
        self.close = close


class KalshiTrainingRow:
    def __init__(self, ticker: str, timestamp: str, probability: float, lookahead_seconds: int, target_btc_price: float = None):
        self.ticker = ticker
        self.timestamp = timestamp
        self.probability = probability
        self.lookahead_seconds = lookahead_seconds
        self.target_btc_price = target_btc_price


class FeatureConfig:
    def __init__(self):
        # Feature flags matching Zig implementation
        self.probability = True
        self.lookahead_seconds = True
    
    def count_features(self):
        count = 0
        if self.probability: count += 1
        if self.lookahead_seconds: count += 1
        return count


def find_most_recent_price_index(timestamp: int, btc_prices: List[BtcPriceRecord]) -> Optional[int]:
    """Find the most recent BTC price whose timestamp is <= the given timestamp."""
    # Extract timestamps for binary search
    timestamps = [p.timestamp for p in btc_prices]
    idx = bisect.bisect_right(timestamps, timestamp) - 1
    if idx < 0:
        return None
    return idx


def parse_and_align(kalshi_path: str, btc_path: str, feature_cfg: FeatureConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse and align Kalshi and BTC datasets."""
    
    # Load Kalshi dataset
    with open(kalshi_path, 'r') as f:
        kalshi_data = json.load(f)
    
    # Extract training rows from the Kalshi dataset structure
    kalshi_rows = []
    for row in kalshi_data['training_rows']:
        kalshi_rows.append(KalshiTrainingRow(
            ticker=row['ticker'],
            timestamp=row['timestamp'],
            probability=row['probability'],
            lookahead_seconds=row['lookahead_seconds'],
            target_btc_price=row.get('target_btc_price')
        ))
    
    # Load BTC dataset
    with open(btc_path, 'r') as f:
        btc_data = json.load(f)
    
    # Extract y_rows from the BTC dataset structure and convert timestamps to Unix seconds
    btc_prices = []
    for candle in btc_data['y_rows']:
        btc_prices.append(BtcPriceRecord(
            timestamp=parse_iso8601_to_seconds(candle['timestamp']),
            close=candle['price']
        ))
    
    # Sort BTC prices by timestamp
    btc_prices.sort(key=lambda x: x.timestamp)
    
    # Align datasets
    aligned_x = []
    aligned_y = []
    
    for row in kalshi_rows:
        kalshi_ts = parse_iso8601_to_seconds(row.timestamp)
        # Add lookahead seconds to get the target timestamp
        target_ts = kalshi_ts + row.lookahead_seconds
        btc_idx = find_most_recent_price_index(target_ts, btc_prices)
        if btc_idx is None:
            continue  # Skip if no matching BTC price
        btc_price = btc_prices[btc_idx].close
        
        # Extract features matching Zig implementation
        features = []
        if feature_cfg.probability:
            features.append(float(row.probability))
        if feature_cfg.lookahead_seconds:
            features.append(float(row.lookahead_seconds))
        
        aligned_x.append(features)
        aligned_y.append(btc_price)
    
    return torch.tensor(aligned_x, dtype=torch.float32), torch.tensor(aligned_y, dtype=torch.float32).unsqueeze(1)


class CryptoForecaster(nn.Module):
    """6-layer MLP for crypto price prediction."""
    
    def __init__(self, num_features: int, hidden1: int, hidden2: int, hidden3: int, 
                 hidden4: int, hidden5: int, hidden6: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, hidden5),
            nn.ReLU(),
            nn.Linear(hidden5, hidden6),
            nn.ReLU(),
            nn.Linear(hidden6, 1),
        )
    
    def forward(self, x):
        return self.network(x)


def train_and_evaluate_crypto_model(kalshi_path: str, btc_path: str):
    """Train and evaluate the crypto price prediction model."""
    
    # Hyperparameters (matching Zig implementation)
    batch_size = 32
    hidden1 = 64  # Reduced since we have fewer features now
    hidden2 = 32
    hidden3 = 16
    hidden4 = 8
    hidden5 = 4
    hidden6 = 2
    learning_rate = 0.001
    epochs = 100
    train_split = 0.8
    
    feature_cfg = FeatureConfig()
    
    print("Loading and aligning crypto datasets...")
    print(f"Reading Kalshi dataset from: {kalshi_path}")
    print(f"Reading BTC dataset from: {btc_path}")
    
    x_tensor, y_tensor = parse_and_align(kalshi_path, btc_path, feature_cfg)
    
    num_samples, num_features = x_tensor.shape
    print(f"Dataset loaded: {num_samples} samples, {num_features} features")
    
    # Split into train/validation
    train_size = int(num_samples * train_split)
    val_size = num_samples - train_size
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Calculate baseline (mean prediction)
    train_mean = y_tensor[:train_size].mean().item()
    
    # Calculate baseline MSE and MAE on validation set
    val_y = y_tensor[train_size:]
    baseline_preds = torch.full_like(val_y, train_mean)
    baseline_mse = torch.mean((baseline_preds - val_y) ** 2).item()
    baseline_mae = torch.mean(torch.abs(baseline_preds - val_y)).item()
    
    print("\n--- Baseline Performance (Mean Prediction) ---")
    print(f"Train mean BTC price: ${train_mean:.2f}")
    print(f"Baseline val_loss (MSE): {baseline_mse:.6f}")
    print(f"Baseline val_mae: ${baseline_mae:.2f}")
    print("----------------------------------------------")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CryptoForecaster(num_features, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6).to(device)
    
    # Initialize optimizer (Adam with same hyperparameters as Zig)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    # Split data
    train_x = x_tensor[:train_size].to(device)
    train_y = y_tensor[:train_size].to(device)
    val_x = x_tensor[train_size:].to(device)
    val_y = y_tensor[train_size:].to(device)
    
    print(f"\nStarting training for {epochs} epochs...")
    
    final_val_mae = 0.0
    first_val_mae = 0.0
    first_val_loss = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Training loop
        for i in range(0, train_size, batch_size):
            batch_end = min(i + batch_size, train_size)
            batch_x = train_x[i:batch_end]
            batch_y = train_y[i:batch_end]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = torch.mean((outputs - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                batch_end = min(i + batch_size, val_size)
                batch_x = val_x[i:batch_end]
                batch_y = val_y[i:batch_end]
                
                outputs = model(batch_x)
                mse = torch.mean((outputs - batch_y) ** 2).item()
                mae = torch.mean(torch.abs(outputs - batch_y)).item()
                
                val_loss += mse
                val_mae += mae
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / val_batches
        
        # Store first epoch metrics
        if epoch == 0:
            first_val_mae = avg_val_mae
            first_val_loss = avg_val_loss
        
        final_val_mae = avg_val_mae
        
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.6f}, val_loss: {avg_val_loss:.6f}, val_mae: ${avg_val_mae:.2f}")
        
        # Show sample predictions on last epoch
        if epoch == epochs - 1:
            print("\n--- Sample Predictions (Validation Set) ---")
            sample_count = min(5, val_size)
            with torch.no_grad():
                for i in range(sample_count):
                    sample_x = val_x[i:i+1]
                    sample_y = val_y[i:i+1]
                    prediction = model(sample_x).item()
                    actual = sample_y.item()
                    diff = prediction - actual
                    print(f"Sample {i + 1}: Predicted: ${prediction:.2f}, Actual: ${actual:.2f}, Error: ${diff:.2f}")
            print("-------------------------------------------")
    
    print("\n--- Final Performance Comparison ---")
    print(f"Baseline val_mae: ${baseline_mae:.2f}")
    print(f"First epoch val_mae: ${first_val_mae:.2f}")
    print(f"Final model val_mae: ${final_val_mae:.2f}")
    
    baseline_improvement = baseline_mae - final_val_mae
    baseline_improvement_pct = (baseline_improvement / baseline_mae) * 100.0
    print(f"Improvement over baseline: ${baseline_improvement:.2f} ({baseline_improvement_pct:.1f}%)")
    
    training_improvement = first_val_mae - final_val_mae
    training_improvement_pct = (training_improvement / first_val_mae) * 100.0
    print(f"Improvement from first epoch: ${training_improvement:.2f} ({training_improvement_pct:.1f}%)")
    print("--------------------------------------")
    print("Training complete!")


if __name__ == "__main__":
    kalshi_path = "kalshi_training_120d_dataset.json"
    btc_path = "btc_price_120d_dataset.json"
    
    train_and_evaluate_crypto_model(kalshi_path, btc_path)
