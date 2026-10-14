# import os
# import pickle
# import streamlit as st
# import torch
# import numpy as np
# import pandas as pd
# from src.model import UserTower
# from src.retriever import FAISSRetriever

# # Page Config
# st.set_page_config(page_title="Two-Tower Recommender", layout="wide")

# # Official MovieLens 1M Occupation Dictionary for the UI
# OCCUPATION_MAP = {
#     0: "other or not specified",
#     1: "academic/educator",
#     2: "artist",
#     3: "clerical/admin",
#     4: "college/grad student",
#     5: "customer service",
#     6: "doctor/health care",
#     7: "executive/managerial",
#     8: "farmer",
#     9: "homemaker",
#     10: "K-12 student",
#     11: "lawyer",
#     12: "programmer",
#     13: "retired",
#     14: "sales/marketing",
#     15: "scientist",
#     16: "self-employed",
#     17: "technician/engineer",
#     18: "tradesman/craftsman",
#     19: "unemployed",
#     20: "writer"
# }


# @st.cache_resource
# def load_resources():
#     encoders = {}
#     # Add 'age' to the loop to load its pickled encoder
#     for feature in ['user_id', 'gender', 'age', 'occupation', 'movie_id', 'year']:
#         with open(f'data/processed/{feature}_encoder.pkl', 'rb') as f:
#             encoders[feature] = pickle.load(f)
            
#     dims = {
#         'user_id': len(encoders['user_id'].classes_),
#         'age': len(encoders['age'].classes_), 
#         'gender': len(encoders['gender'].classes_),
#         'occupation': len(encoders['occupation'].classes_)
#     }
    
#     user_tower = UserTower(dims, embedding_dim=32, output_dim=128)
#     user_tower.load_state_dict(torch.load('artifacts/user_tower.pth', weights_only=True))
#     user_tower.eval()
    
#     retriever = FAISSRetriever()
#     retriever.load_index()
    
#     # Update metadata loader to include genres
#     movies_df = pd.read_csv(
#         'data/raw/movies.dat', sep='::', encoding='latin-1', engine='python',
#         names=['movie_id', 'title', 'genres'],
#         usecols=['movie_id', 'title', 'genres']
#     )
    
#     return encoders, user_tower, retriever, movies_df

# # Initialize resources
# # Initialize resources
# encoders, user_tower, retriever, movies_df = load_resources()

# st.title("🎬 Sub-Millisecond Movie Recommendations")
# st.markdown("Powered by a PyTorch Two-Tower Architecture & FAISS Vector Search")

# # Sidebar for Real-Time User Input
# with st.sidebar:
#     st.header("Simulate User Profile")
    
#     input_user_id = st.number_input("User ID (1-6040)", min_value=1, max_value=6040, value=42)
#     input_gender = st.selectbox("Gender", ['M', 'F'])
    
#     # 1M dataset pre-defined age buckets
#     age_mapping = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
#     selected_age_label = st.selectbox("Age Group", list(age_mapping.values()))
#     # Reverse lookup to get the integer key (1, 18, 25...)
#     input_age = [k for k, v in age_mapping.items() if v == selected_age_label][0]
    
#     # --- MODIFIED SECTION ---
#     # Display the string values from OCCUPATION_MAP in the dropdown
#     selected_occ_label = st.selectbox("Occupation", list(OCCUPATION_MAP.values()))
#     # Reverse lookup to get the original integer ID (0-20)
#     input_occupation = [k for k, v in OCCUPATION_MAP.items() if v == selected_occ_label][0]
#     # ------------------------
    
#     generate_btn = st.button("Generate Feed", type="primary", use_container_width=True)

# if generate_btn:
#     with st.spinner("Executing Vector Search..."):
#         encoded_user_id = encoders['user_id'].transform([input_user_id])[0]
#         encoded_gender = encoders['gender'].transform([input_gender])[0]
#         encoded_occ = encoders['occupation'].transform([input_occupation])[0]
#         # Use the loaded encoder instead of pd.cut
#         encoded_age = encoders['age'].transform([input_age])[0]

#         user_tensor = torch.tensor([[encoded_user_id, encoded_age, encoded_gender, encoded_occ]], dtype=torch.long)
        
#         with torch.no_grad():
#             user_vector = user_tower(user_tensor).numpy()
            
#         scores, internal_movie_indices = retriever.search(user_vector, top_k=10)
#         actual_movie_ids = encoders['movie_id'].inverse_transform(internal_movie_indices)
#         recommended_movies = movies_df[movies_df['movie_id'].isin(actual_movie_ids)]
        
#         # Render Results with Genres
#         st.subheader("Top 10 Recommendations")
#         cols = st.columns(2)
#         for i, row in enumerate(recommended_movies.itertuples()):
#             col_idx = i % 2
#             with cols[col_idx]:
#                 clean_genres = row.genres.replace('|', ', ')
#                 st.info(f"**{row.title}**\n\n🎭 Genres: {clean_genres}")


import os
import pickle
import time
import streamlit as st
import torch
import numpy as np
import pandas as pd
from src.model import UserTower
from src.retriever import FAISSRetriever

# Page Config
st.set_page_config(page_title="Two-Tower Recommender", layout="wide")

# Official MovieLens 1M Occupation Dictionary for the UI
OCCUPATION_MAP = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}


@st.cache_resource
def load_resources():
    encoders = {}
    # Add 'age' to the loop to load its pickled encoder
    for feature in ['user_id', 'gender', 'age', 'occupation', 'movie_id', 'year']:
        with open(f'data/processed/{feature}_encoder.pkl', 'rb') as f:
            encoders[feature] = pickle.load(f)
            
    dims = {
        'user_id': len(encoders['user_id'].classes_),
        'age': len(encoders['age'].classes_), 
        'gender': len(encoders['gender'].classes_),
        'occupation': len(encoders['occupation'].classes_)
    }
    
    user_tower = UserTower(dims, embedding_dim=32, output_dim=128)
    user_tower.load_state_dict(torch.load('artifacts/user_tower.pth', weights_only=True))
    user_tower.eval()
    
    retriever = FAISSRetriever()
    retriever.load_index()
    
    # Update metadata loader to include genres
    movies_df = pd.read_csv(
        'data/raw/movies.dat', sep='::', encoding='latin-1', engine='python',
        names=['movie_id', 'title', 'genres'],
        usecols=['movie_id', 'title', 'genres']
    )
    
    return encoders, user_tower, retriever, movies_df

# Initialize resources
encoders, user_tower, retriever, movies_df = load_resources()

st.title("🎬 Sub-Millisecond Movie Recommendations")
st.markdown("Powered by a PyTorch Two-Tower Architecture & FAISS Vector Search")

# Sidebar for Real-Time User Input
with st.sidebar:
    st.header("Simulate User Profile")
    
    input_user_id = st.number_input("User ID (1-6040)", min_value=1, max_value=6040, value=42)
    input_gender = st.selectbox("Gender", ['M', 'F'])
    
    # 1M dataset pre-defined age buckets
    age_mapping = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
    selected_age_label = st.selectbox("Age Group", list(age_mapping.values()))
    # Reverse lookup to get the integer key (1, 18, 25...)
    input_age = [k for k, v in age_mapping.items() if v == selected_age_label][0]
    
    # --- MODIFIED SECTION ---
    # Display the string values from OCCUPATION_MAP in the dropdown
    selected_occ_label = st.selectbox("Occupation", list(OCCUPATION_MAP.values()))
    # Reverse lookup to get the original integer ID (0-20)
    input_occupation = [k for k, v in OCCUPATION_MAP.items() if v == selected_occ_label][0]
    # ------------------------
    
    generate_btn = st.button("Generate Feed", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner("Executing Vector Search..."):
        
        # --- START OVERALL TIMER ---
        start_time = time.time()
        
        encoded_user_id = encoders['user_id'].transform([input_user_id])[0]
        encoded_gender = encoders['gender'].transform([input_gender])[0]
        encoded_occ = encoders['occupation'].transform([input_occupation])[0]
        # Use the loaded encoder instead of pd.cut
        encoded_age = encoders['age'].transform([input_age])[0]

        user_tensor = torch.tensor([[encoded_user_id, encoded_age, encoded_gender, encoded_occ]], dtype=torch.long)
        
        # --- MEASURE PYTORCH LATENCY ---
        pt_start = time.time()
        with torch.no_grad():
            user_vector = user_tower(user_tensor).numpy()
        pt_end = time.time()
            
        # --- MEASURE FAISS LATENCY ---
        faiss_start = time.time()
        scores, internal_movie_indices = retriever.search(user_vector, top_k=10)
        faiss_end = time.time()
        
        # --- END OVERALL TIMER ---
        end_time = time.time()
        
        # Calculate all metrics in milliseconds
        total_ms = (end_time - start_time) * 1000
        pt_ms = (pt_end - pt_start) * 1000
        faiss_ms = (faiss_end - faiss_start) * 1000

        actual_movie_ids = encoders['movie_id'].inverse_transform(internal_movie_indices)
        recommended_movies = movies_df[movies_df['movie_id'].isin(actual_movie_ids)]
        
        # --- RENDER TIMING METRICS TO UI ---
        st.success(f"⚡ **Total Inference:** {total_ms:.2f} ms | **PyTorch Tower:** {pt_ms:.2f} ms | **FAISS Search:** {faiss_ms:.2f} ms")
        
        # Render Results with Genres
        st.subheader("Top 10 Recommendations")
        cols = st.columns(2)
        for i, row in enumerate(recommended_movies.itertuples()):
            col_idx = i % 2
            with cols[col_idx]:
                clean_genres = row.genres.replace('|', ', ')
                st.info(f"**{row.title}**\n\n🎭 Genres: {clean_genres}")