import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import TwoTowerRecommender

def train_model():
    # 1. Setup Environment and Load Data
    os.makedirs('artifacts', exist_ok=True)
    
    # Increased batch size to 2048 to speed up MovieLens 1M training on CPU
    train_dl, val_dl, dims = get_dataloaders(batch_size=2048)
    
    # 2. Initialize Model, Optimizer, and Loss Function
    model = TwoTowerRecommender(dims)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # BCEWithLogitsLoss automatically applies Sigmoid to our dot products
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    print("Starting Training on MovieLens 1M...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for user_batch, item_batch, label_batch in train_dl:
            # --- DYNAMIC NEGATIVE SAMPLING ---
            negative_item_batch = item_batch[torch.randperm(item_batch.size(0))]
            negative_labels = torch.zeros_like(label_batch)

            combined_users = torch.cat([user_batch, user_batch], dim=0)
            combined_items = torch.cat([item_batch, negative_item_batch], dim=0)
            combined_labels = torch.cat([label_batch, negative_labels], dim=0)

            # --- OPTIMIZE WITH LOSS ---
            optimizer.zero_grad()
            predictions = model(combined_users, combined_items)
            
            loss = criterion(predictions, combined_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(train_dl):.4f}")

    # --- DECOUPLE AND EXPORT ---
    print("\nDecoupling towers and exporting artifacts...")

    torch.save(model.user_tower.state_dict(), 'artifacts/user_tower.pth')

    model.eval()
    with torch.no_grad():
        all_items = torch.unique(train_dl.dataset.item_features, dim=0)
        item_embeddings = model.item_tower(all_items)
        np.save('artifacts/item_embeddings.npy', item_embeddings.numpy())
        
    print("Export complete. Files saved to artifacts/. Ready for Phase 4 (FAISS).")

if __name__ == "__main__":
    train_model()