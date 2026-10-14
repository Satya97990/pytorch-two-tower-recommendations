# 🎬 Sub-Millisecond Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%2B-Learning-orange.svg)]()
[![FAISS](https://img.shields.io/badge/Vector%20Database-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-red.svg)](https://streamlit.io/)

A production-ready, deep learning-based recommendation system that predicts user movie preferences and serves real-time results using a vector database. Built using a decoupled Two-Tower architecture and the MovieLens 1M dataset, this application is optimized for low-latency inference and deployed via Streamlit Community Cloud.

### 🚀 Live Demo
**[Click here to view the live application!](https://pytorch-two-tower-recommendations-29tv9ggspgtbeafzb5bwqz.streamlit.app/)**  


---



---

## 🏗️ System Architecture & Key Features

This project emphasizes clean Object-Oriented Programming (OOP), modular design, and efficient production serving:

*   **Deep Learning Two-Tower Model:** Learns continuous vector representations (embeddings) for both users and movies based on over 1,000,000 implicit interactions.
*   **Decoupled Inference:** To maximize speed, the item tower is pre-computed. Only the user tower runs dynamically during a live request.
*   **Sub-Millisecond Vector Search:** Integrates **FAISS (Facebook AI Similarity Search)** as a vector database to index over 4,000 high-dimensional movie embeddings, bypassing brute-force computation for instant nearest-neighbor retrieval.
*   **Automated Cloud Availability:** Deployed on Streamlit Community Cloud with a custom **GitHub Actions CI/CD workflow** (`wake.yml`) that pings the server every 4 hours to prevent the app from hibernating.

---

## 🛠️ Tech Stack
*   **Core:** Python
*   **Data Processing:** Pandas, Scikit-learn (`LabelEncoder`)
*   **Vector Database (Retrieval):** FAISS (CPU)
*   **Web Framework & Deployment:** Streamlit, GitHub Actions

---

## 📂 Project Structure

```text
two-tower-recsys/
 ┣ .github/
 ┃ ┗ workflows/
 ┃   ┗ wake.yml             # Automated ping to keep the cloud app awake
 ┣ artifacts/               # Compiled model weights and FAISS index
 ┃ ┣ faiss_index.bin
 ┃ ┣ item_embeddings.npy
 ┃ ┗ user_tower.pth
 ┣ data/
 ┃ ┣ processed/             # Pickled LabelEncoders for consistent UI mapping
 ┃ ┗ raw/                   # MovieLens 1M dataset (.dat files)
 ┣ src/                     # OOP Modular Pipeline
 ┃ ┣ data_loader.py         # ETL and binarization logic
 ┃ ┣ model.py               # Neural network architecture definition
 ┃ ┣ retriever.py           # FAISS indexing and search logic
 ┃ ┗ train.py               # Training loop and artifact decoupling
 ┣ app.py                   # Streamlit frontend UI
 ┣ requirements.txt         # Production dependencies
 ┗ README.md
