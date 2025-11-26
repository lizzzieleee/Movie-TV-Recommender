# Movie & TV Recommender
A content-based recommendation engine powered by TMDB metadata, Sentence Transformers, and FAISS — now with an offline, fully preloaded corpus (1950–2025).

---

## Overview
This project builds a high-quality movie and TV recommender using metadata from The Movie Database (TMDB) and modern embedding techniques. Users select titles they like, and the app returns similar movies/TV shows using cosine similarity over semantic text embeddings.

The app runs in Streamlit and is designed to showcase practical recommendation-system skills for entertainment-tech roles.

---

## Key Features
- TMDB as the source of metadata (overview, genres, cast, crew, keywords)
- Rich feature engineering combining multiple content signals
- Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings
- FAISS for fast, scalable vector similarity search
- Preloaded corpus covering movies and TV from the 1950s through 2025
- Fully functional offline mode (no TMDB API key required)
- Streamlit front-end with search, title selection, and detailed recommendation cards

---

## Architecture

### 1. Data Ingestion (TMDB API)
The corpus is originally built using TMDB’s `/discover` and `/details` endpoints:
- Movies and TV shows from 1950–2025
- Fields collected: title, overview, genres, keywords, cast, crew, release dates, popularity, poster paths

### 2. Feature Construction
Each title is converted into a rich text representation:

```
overview + (genres × 3) + (keywords × 2) + cast + crew
```

### 3. Embedding Generation
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings normalized for cosine-similarity FAISS search

### 4. Indexing
- FAISS `IndexFlatIP` for cosine similarity
- Saved to:
  - `models/index.faiss`
  - `models/embeddings.npy`
  - `models/meta.parquet`
- Falls back to `sklearn`’s `NearestNeighbors` if FAISS unavailable

### 5. Recommendation
Given selected titles:
- Build a combined query embedding
- Retrieve nearest neighbors from FAISS index
- Blend similarity with popularity (optional)
- Display results with title, metadata, score, and poster

---

## Offline Mode (No TMDB API Key Required)
The repository includes a fully precomputed corpus and FAISS index.

In offline mode:
- No TMDB API key is required
- Search is performed against the local metadata
- Recommendations run entirely from precomputed embeddings
- Rich details (cast, crew, keywords) are limited without a live API call
- Posters are displayed using stored TMDB image paths when available

This makes the project easy to demo for recruiters or colleagues.

---

## Running the App

### Option A — Offline Mode (recommended)

Clone and run:

```bash
git clone https://github.com/lizzzieleee/Movie-TV-Recommender.git
cd Movie-TV-Recommender
pip install -r requirements.txt
streamlit run app.py
```

No `.env` file or TMDB key needed.

---

### Option B — Online Mode (full TMDB features)

Create a `.env` file in the project root:

```bash
TMDB_API_KEY=your_tmdb_key_here
```

Or set the key in Streamlit secrets.

This enables:
- TMDB search autocomplete  
- Full cast/crew/keyword panels  
- The ability to rebuild or extend the corpus  

---

## Project Structure

```
Movie-TV-Recommender/
│
├── app.py                  # Streamlit front-end + recommender logic
├── build_corpus.py         # Optional script to rebuild the corpus
│
├── models/
│   ├── meta.parquet        # Preloaded metadata (1950–2025)
│   ├── index.faiss         # Precomputed FAISS index
│   ├── embeddings.npy      # Precomputed embeddings
│   └── nn.joblib           # sklearn index fallback
│
├── requirements.txt
└── README.md
```

---

## Contact
If you are interested in recommendation systems, content analytics, or entertainment-tech data roles, feel free to reach out.
