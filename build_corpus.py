# build_corpus.py
"""
One-time offline builder for the TMDB recommender.

Usage:
    python build_corpus.py
"""

from app import quick_build_corpus, build_index_and_save

def main():
    # Heavier settings than you'd use inside Streamlit
    pages_movie = 80   # tweak as you like
    pages_tv = 60
    min_vote_count = 200

    print("Building corpus…")
    enriched = quick_build_corpus(
        pages_movie=pages_movie,
        pages_tv=pages_tv,
        min_vote_count=min_vote_count,
    )
    print(f"Collected {len(enriched)} enriched titles")

    print("Building index and saving to models/…")
    build_index_and_save(enriched)

    print("Done! Check the models/ directory for:")
    print("  - meta.parquet")
    print("  - index.faiss or nn.joblib")
    print("  - embeddings.npy")

if __name__ == "__main__":
    main()

