import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, dims, embedding_dim=32, output_dim=128):
        super().__init__()
        # Initialize embeddings based on dynamic dictionary sizes from data_loader
        self.user_emb = nn.Embedding(dims['user_id'], embedding_dim)
        self.age_emb = nn.Embedding(dims['age'], embedding_dim)
        self.gender_emb = nn.Embedding(dims['gender'], embedding_dim)
        self.occ_emb = nn.Embedding(dims['occupation'], embedding_dim)
        
        # 4 features concatenated
        input_dim = embedding_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, user_features):
        # Slice the incoming [batch_size, 4] tensor
        u_emb = self.user_emb(user_features[:, 0])
        a_emb = self.age_emb(user_features[:, 1])
        g_emb = self.gender_emb(user_features[:, 2])
        o_emb = self.occ_emb(user_features[:, 3])
        
        concat_embs = torch.cat([u_emb, a_emb, g_emb, o_emb], dim=1)
        return self.mlp(concat_embs)

class ItemTower(nn.Module):
    def __init__(self, dims, embedding_dim=32, output_dim=128):
        super().__init__()
        self.movie_emb = nn.Embedding(dims['movie_id'], embedding_dim)
        self.year_emb = nn.Embedding(dims['year'], embedding_dim)
        
        # 2 features concatenated
        input_dim = embedding_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, item_features):
        # Slice the incoming [batch_size, 2] tensor
        m_emb = self.movie_emb(item_features[:, 0])
        y_emb = self.year_emb(item_features[:, 1])
        
        concat_embs = torch.cat([m_emb, y_emb], dim=1)
        return self.mlp(concat_embs)

class TwoTowerRecommender(nn.Module):
    def __init__(self, dims, embedding_dim=32, output_dim=128):
        super().__init__()
        self.user_tower = UserTower(dims, embedding_dim, output_dim)
        self.item_tower = ItemTower(dims, embedding_dim, output_dim)
        
    def forward(self, user_features, item_features):
        user_vector = self.user_tower(user_features)
        item_vector = self.item_tower(item_features)
        
        # Compute the dot product across the 128-D embedding space
        # Shape: [batch_size, 128] * [batch_size, 128] -> sum -> [batch_size]
        scores = (user_vector * item_vector).sum(dim=1)
        return scores

if __name__ == "__main__":
    from data_loader import get_dataloaders
    
    # 1. Fetch dummy data to test the architecture
    train_dl, _, dims = get_dataloaders(batch_size=5)
    user_batch, item_batch, label_batch = next(iter(train_dl))
    
    # 2. Initialize the model
    model = TwoTowerRecommender(dims)
    
    # 3. Test the forward pass
    predictions = model(user_batch, item_batch)
    
    print("User Vector Shape (Internal):", model.user_tower(user_batch).shape)
    print("Item Vector Shape (Internal):", model.item_tower(item_batch).shape)
    print("Final Output Scores Shape:", predictions.shape)