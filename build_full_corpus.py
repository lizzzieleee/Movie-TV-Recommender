# build_full_corpus.py
from app import quick_build_corpus, build_index_and_save

def main():
    enriched = quick_build_corpus(
        pages_movie=80,      # heavier build
        pages_tv=60,
        min_vote_count=200,
    )
    print(f"Collected {len(enriched)} titles")
    build_index_and_save(enriched)
    print("Index + meta saved in models/")

if __name__ == "__main__":
    main()

