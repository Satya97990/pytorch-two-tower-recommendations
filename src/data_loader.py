import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MovieLensProcessor:
    def __init__(self, raw_path='data/raw', processed_path='data/processed'):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.encoders = {}
        os.makedirs(self.processed_path, exist_ok=True)

    def load_and_merge_data(self):
        # 1. Load Interactions (ratings.dat)
        ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(f'{self.raw_path}/ratings.dat', sep='::', names=ratings_cols, encoding='latin-1', engine='python')

        # 2. Load Users (users.dat)
        user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        users = pd.read_csv(f'{self.raw_path}/users.dat', sep='::', names=user_cols, encoding='latin-1', engine='python')

        # 3. Load Movies (movies.dat)
        movie_cols = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(f'{self.raw_path}/movies.dat', sep='::', names=movie_cols, encoding='latin-1', engine='python')

        # Extract Release Year as a feature from the title (e.g., "Toy Story (1995)")
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies['year'] = movies['year'].fillna(movies['year'].median()).astype(int)

        # Merge into a single flat dataframe
        df = ratings.merge(users, on='user_id').merge(movies[['movie_id', 'year']], on='movie_id')
        
        # 4. Implicit Formulation: Ratings >= 4 are 1 (Positive), else 0 (Negative)
        df['label'] = (df['rating'] >= 4).astype(float)
        
        return df

    def encode_features(self, df):
        # Categorical features to map to contiguous integers for nn.Embedding
        # ADDED 'age' here because MovieLens 1M pre-buckets ages.
        categorical_features = ['user_id', 'gender', 'age', 'occupation', 'movie_id', 'year']
        
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            self.encoders[feature] = le
            
            # Save the encoder for Phase 5 (Streamlit Inference)
            with open(f'{self.processed_path}/{feature}_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
                
        # REMOVED the pd.cut() logic here because 1M data is already bucketed.
        
        return df

class TwoTowerDataset(Dataset):
    def __init__(self, df):
        # User Features Tensor
        self.user_features = torch.tensor(
            df[['user_id', 'age', 'gender', 'occupation']].values, dtype=torch.long
        )
        # Item Features Tensor
        self.item_features = torch.tensor(
            df[['movie_id', 'year']].values, dtype=torch.long
        )
        # Labels Tensor
        self.labels = torch.tensor(
            df['label'].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_features[idx], self.item_features[idx], self.labels[idx]

def get_dataloaders(batch_size=1024):
    processor = MovieLensProcessor()
    df = processor.load_and_merge_data()
    df = processor.encode_features(df)
    
    # Simple 80/20 sequential split for debugging
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    train_dataset = TwoTowerDataset(train_df)
    val_dataset = TwoTowerDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Return encoders dictionary sizes to dynamically size our PyTorch Embedding layers later
    feature_dims = {
        'user_id': len(processor.encoders['user_id'].classes_),
        'age': len(processor.encoders['age'].classes_), # UPDATED to read dynamically from the 7 buckets
        'gender': len(processor.encoders['gender'].classes_),
        'occupation': len(processor.encoders['occupation'].classes_),
        'movie_id': len(processor.encoders['movie_id'].classes_),
        'year': len(processor.encoders['year'].classes_)
    }
    
    return train_loader, val_loader, feature_dims

if __name__ == "__main__":
    # Test the pipeline locally
    train_dl, val_dl, dims = get_dataloaders(batch_size=5)
    
    print("Feature Vocabulary Sizes:", dims)
    
    for user_batch, item_batch, label_batch in train_dl:
        print("\nBatch shapes:")
        print("User Features:", user_batch.shape) # Expected: [5, 4]
        print("Item Features:", item_batch.shape) # Expected: [5, 2]
        print("Labels:", label_batch.shape)       # Expected: [5]
        break