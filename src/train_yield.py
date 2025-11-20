import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_yield import SuzukiYieldDataset
from model_yield import SuzukiYieldGNN

def train_model(csv_path, epochs=100, batch_size=32, lr=5e-4, device='cpu'):
    """
    Train the Suzuki yield prediction model with conditions (ligand/base/solvent)
    
    Key improvements:
    - Yield normalization to 0-1 scale
    - Condition integration
    - Early stopping with patience
    - Learning rate scheduling
    - Gradient clipping
    - Proper train/val/test split
    """
    # Load dataset
    dataset = SuzukiYieldDataset(csv_path)
    
    # Train/val/test split (70/15/15)
    n = len(dataset)
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Get dimensions from first sample
    sample = dataset[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
    cond_dim = sample.conditions.shape[1]  # Use dim 1 since it's now 2D [1, 32]
    
    # Initialize model
    model = SuzukiYieldGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        cond_dim=cond_dim,
        hidden_dim=256,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Was 10, now 5
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch).squeeze(-1)
            target = batch.y.squeeze(-1)
            
            # Compute loss
            loss = criterion(pred, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * target.size(0)
            train_samples += target.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze(-1)
                target = batch.y.squeeze(-1)
                
                loss = criterion(pred, target)
                val_loss += loss.item() * target.size(0)
                val_samples += target.size(0)
        
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples
        
        # Convert to percentage scale for reporting (since yields were normalized to 0-1)
        train_mse_pct = avg_train_loss * 10000  # (0-1 scale)^2 * 100^2 = percentage^2
        val_mse_pct = avg_val_loss * 10000
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train MSE (% scale): {train_mse_pct:.2f}, RMSE: {np.sqrt(train_mse_pct):.2f}%")
        print(f"  Val MSE (% scale): {val_mse_pct:.2f}, RMSE: {np.sqrt(val_mse_pct):.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pt')
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print("Early stopping triggered")
                break
    
    # Load best model and test
    print("\n" + "="*50)
    print("Loading best model for testing...")
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']}")
    
    model.eval()
    test_loss = 0
    test_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)
            target = batch.y.squeeze(-1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            loss = criterion(pred, target)
            test_loss += loss.item() * target.size(0)
            test_samples += target.size(0)
    
    test_mse = (test_loss / test_samples) * 10000
    
    # Calculate additional metrics
    all_preds = np.array(all_preds) * 100  # Convert back to percentage
    all_targets = np.array(all_targets) * 100
    
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = 1 - np.sum((all_targets - all_preds)**2) / np.sum((all_targets - all_targets.mean())**2)
    
    # Mean absolute percentage error (for yields > 5%)
    mask = all_targets > 5
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    else:
        mape = None
    
    print("\n" + "="*50)
    print("=== Final Test Results ===")
    print("="*50)
    print(f"Test MSE (% scale): {test_mse:.2f}")
    print(f"Test RMSE (% scale): {np.sqrt(test_mse):.2f}%")
    print(f"Test MAE: {mae:.2f}%")
    print(f"Test R²: {r2:.4f}")
    if mape is not None:
        print(f"Test MAPE (yields > 5%): {mape:.2f}%")
    print("="*50)
    
    # Show some example predictions
    print("\nExample predictions (first 10):")
    print("Predicted | Actual")
    print("-" * 20)
    for i in range(min(10, len(all_preds))):
        print(f"{all_preds[i]:6.1f}%  | {all_targets[i]:6.1f}%")
    
    return model

if __name__ == "__main__":
    csv_path = "data/processed/suzuki_products.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = train_model(
        csv_path=csv_path,
        epochs=150,
        batch_size=32,
        lr=1e-4,
        device=device
    )