import torch
from torch_geometric.loader import DataLoader
from dataset_yield import SuzukiYieldDataset
from model_yield import SuzukiYieldGNN

def train_model(csv_path, epochs=30, batch_size=32, lr=1e-3, device='cpu'):
    ds = SuzukiYieldDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    sample = ds[0]
    in_dim = sample.x.shape[1]

    model = SuzukiYieldGNN(in_dim=in_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        n = 0
        for data in loader:
            data = data.to(device)
            pred = model(data).squeeze(-1)   # remove the last dimension so pred is [batch_size]
            target = data.y.squeeze(-1)      # ensure target is also [batch_size]
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*target.size(0)
            n += target.size(0)
        avg_loss = total_loss / n
        print(f"Epoch {epoch}/{epochs} — MSE Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    csv_path = "data/processed/suzuki_products.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(csv_path, epochs=30, batch_size=32, lr=1e-3, device=device)
