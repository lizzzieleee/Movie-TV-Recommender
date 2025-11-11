# app.py
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

import sys
import platform
import torch 
import sentence_transformers

# ==============================
# Env + constants (v3 API key)
# ==============================
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

# --- Prefer Streamlit secrets when available ---
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY")

# --- Fallback for local development ---
if not TMDB_API_KEY:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("⚠️ TMDB_API_KEY not found. Add it to Streamlit Secrets or .env.")
    st.stop()
    
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p"  # w92 | w154 | w185 | w342 | w500 | original

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.sidebar.write("Python:", sys.version)
try:
    import faiss
    st.sidebar.write("FAISS:", faiss.__version__)
except Exception as e:
    st.sidebar.write("FAISS: not loaded", str(e))
st.sidebar.write("Torch:", torch.__version__)
st.sidebar.write("SentenceTransformers:", sentence_transformers.__version__)
st.sidebar.write("NumPy:", np.__version__)
st.sidebar.write("Pandas:", pd.__version__)

# ==============================
# Heavy libs for embeddings/index
# ==============================
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    st.error("Missing libraries. Install with:\n\npip install sentence-transformers faiss-cpu")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EMBEDDER = get_embedder()

# ==============================
# HTTP helpers (v3 key in params)
# ==============================
def get_tmdb(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(params or {})
    params["api_key"] = TMDB_API_KEY
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# --- Helper to build full poster URLs ---
def poster_url(poster_path: str | None, size: str = "w342") -> str | None:
    """Return the full TMDB image URL for a given poster path."""
    return f"{IMG_BASE}/{size}{poster_path}" if poster_path else None

# ==============================
# Rich-detail fetchers (genres/keywords/credits)
# ==============================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_details_rich(media_type: str, tmdb_id: int) -> Dict[str, Any]:
    """Full details including overview, genres, keywords, cast & crew."""
    details = get_tmdb(f"/{media_type}/{tmdb_id}")
    keywords_data = get_tmdb(f"/{media_type}/{tmdb_id}/keywords")
    credits_data = get_tmdb(f"/{media_type}/{tmdb_id}/credits")

    genres = [g["name"] for g in details.get("genres", [])]
    kw = keywords_data.get("keywords") or keywords_data.get("results") or []
    keywords = [k["name"] for k in kw]

    cast = [c["name"] for c in (credits_data.get("cast") or []) if c.get("name")][:10]
    crew = [c["name"] for c in (credits_data.get("crew") or [])
            if c.get("job") in {"Director", "Writer", "Screenplay"} and c.get("name")][:5]

    return {
        "title": details.get("title") or details.get("name") or "",
        "overview": details.get("overview", "") or "",
        "genres": genres,
        "keywords": keywords,
        "cast": cast,
        "crew": crew,
        "poster_path": details.get("poster_path"),
        "release_date": details.get("release_date") or details.get("first_air_date") or "",
        "runtime": details.get("runtime"),
        "seasons": details.get("number_of_seasons"),
        "episodes": details.get("number_of_episodes"),
        "vote_average": details.get("vote_average"),
        "popularity": details.get("popularity"),
        "homepage": details.get("homepage"),
        "status": details.get("status"),
    }

def make_feature_text_rich(data: Dict[str, Any]) -> str:
    """Overview + 3×Genres + 2×Keywords + Cast + Crew."""
    parts = [
        data.get("overview", ""),
        " ".join(data.get("genres", [])) * 3,
        " ".join(data.get("keywords", [])) * 2,
        " ".join(data.get("cast", [])),
        " ".join(data.get("crew", [])),
    ]
    return " ".join([p for p in parts if p]).strip()

# ==============================
# Discover API to build a corpus
# ==============================
def collect_discover(media_type="movie", pages=3, date_gte="2016-01-01", language="en") -> pd.DataFrame:
    rows = []
    date_field = "primary_release_date.gte" if media_type == "movie" else "first_air_date.gte"
    for p in range(1, pages + 1):
        j = get_tmdb(
            f"/discover/{media_type}",
            {
                date_field: date_gte,
                "sort_by": "popularity.desc",
                "include_adult": False,
                "with_original_language": language,
                "page": p,
            },
        )
        for r in j.get("results", []):
            rows.append({
                "tmdb_id": int(r["id"]),
                "media_type": media_type,
                "title": r.get("title") or r.get("name") or "",
                "overview": (r.get("overview") or "").strip(),
                "popularity": float(r.get("popularity") or 0.0),
                "release_date": r.get("release_date") or r.get("first_air_date") or None,
                "poster_path": r.get("poster_path"),
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=True)
def quick_build_corpus(pages_movie=4, pages_tv=3) -> pd.DataFrame:
    """Quick corpus (~150–250 items) with rich feature text assembled."""
    movies = collect_discover("movie", pages=pages_movie, date_gte="2016-01-01")
    tv = collect_discover("tv", pages=pages_tv, date_gte="2016-01-01")
    pool = pd.concat([movies, tv], ignore_index=True).drop_duplicates(["media_type", "tmdb_id"])

    rows = []
    for _, row in pool.iterrows():
        try:
            rich = fetch_details_rich(row["media_type"], int(row["tmdb_id"]))
            rich["media_type"] = row["media_type"]
            rich["tmdb_id"] = row["tmdb_id"]
            # Prefer details overview if present; fallback to discover
            rich["overview"] = rich.get("overview") or (row.get("overview") or "")
            rich["title"] = row.get("title") or rich.get("title") or ""
            rich["poster_path"] = rich.get("poster_path") or row.get("poster_path")
            rich["release_date"] = row.get("release_date") or rich.get("release_date")
            rich["popularity"] = row.get("popularity", 0.0)
            rich["feature_text"] = make_feature_text_rich(rich)
            rows.append(rich)
        except Exception as e:
            print("[skip]", row.get("title"), e)
            continue

    enriched = pd.DataFrame(rows)
    return enriched

# ==============================
# Index build/load + recommend
# ==============================
def build_index_and_save(enriched: pd.DataFrame):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    texts = enriched["feature_text"].fillna("").tolist()
    X = EMBEDDER.encode(texts, normalize_embeddings=True).astype("float32")

    # Always save meta
    meta_cols = ["media_type","tmdb_id","title","release_date","popularity","poster_path"]
    enriched[meta_cols].to_parquet(MODELS_DIR / "meta.parquet", index=False)

    if USE_FAISS:
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(MODELS_DIR / "index.faiss"))
        np.save(MODELS_DIR / "embeddings.npy", X)  # optional
        return ("faiss", index), X
    else:
        nn = NearestNeighbors(metric="cosine").fit(X)
        joblib.dump(nn, MODELS_DIR / "nn.joblib")
        np.save(MODELS_DIR / "embeddings.npy", X)
        return ("sklearn", nn), X

def load_index_and_meta():
    meta_path = MODELS_DIR / "meta.parquet"
    if not meta_path.exists():
        return pd.DataFrame(), None, None

    meta = pd.read_parquet(meta_path)
    faiss_path = MODELS_DIR / "index.faiss"
    nn_path = MODELS_DIR / "nn.joblib"
    emb_path = MODELS_DIR / "embeddings.npy"
    X = np.load(emb_path) if emb_path.exists() else None

    if faiss_path.exists():
        # FAISS backend
        index = faiss.read_index(str(faiss_path))
        return meta, ("faiss", index), X
    if nn_path.exists():
        # sklearn backend
        from sklearn.neighbors import NearestNeighbors
        import joblib
        nn = joblib.load(nn_path)
        return meta, ("sklearn", nn), X

    return meta, None, X

def feature_text_for_id(media_type: str, tmdb_id: int) -> str:
    rich = fetch_details_rich(media_type, tmdb_id)
    return make_feature_text_rich(rich)

def query_vector_from_selected(liked_items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Builds a query vector directly from selected items (exact IDs)."""
    if not liked_items:
        return None
    texts = []
    for it in liked_items[:5]:  # up to 5
        try:
            txt = feature_text_for_id(it["media_type"], int(it["tmdb_id"]))
            if txt:
                texts.append(txt)
        except Exception:
            pass
    if not texts:
        return None
    V = EMBEDDER.encode(texts, normalize_embeddings=True).astype("float32")
    return V.mean(axis=0, keepdims=True)

def recommend_from_selected(
    liked_items, meta, index_tuple, X, k=12, popularity_blend=0.2
):
    # Build query vector
    qvec = query_vector_from_selected(liked_items)
    if qvec is None or index_tuple is None or meta.empty:
        return pd.DataFrame(columns=list(meta.columns) + ["score"])

    backend, index = index_tuple
    m = max(3 * k, 60)

    if backend == "faiss":
        D, I = index.search(qvec.astype("float32"), m)
        sims = D[0]
        idxs = I[0]
    else:
        # sklearn NearestNeighbors (cosine distance -> similarity = 1 - dist)
        from sklearn.neighbors import NearestNeighbors  # safe if fallback
        dists, idxs = index.kneighbors(qvec, n_neighbors=min(m, X.shape[0]))
        sims = 1.0 - dists[0]
        idxs = idxs[0]

    recs = meta.iloc[idxs].copy().reset_index(drop=True)

    # drop liked items
    liked_set = {(it["media_type"], int(it["tmdb_id"])) for it in liked_items}
    recs = recs[~recs.apply(lambda r: (r["media_type"], int(r["tmdb_id"])) in liked_set, axis=1)]

    # normalize similarity
    sims = sims[: len(recs)]
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # optional popularity blend
    if popularity_blend and "popularity" in recs.columns:
        pop = recs["popularity"].to_numpy()
        pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)
        recs["score"] = (1 - popularity_blend) * sims + popularity_blend * pop
        recs = recs.sort_values("score", ascending=False)
    else:
        recs["score"] = sims

    return recs.head(k).reset_index(drop=True)

# ==============================
# UI (Recommender-only) with type-ahead + chips
# ==============================
st.set_page_config(page_title="🎬 TMDB Recommender", page_icon="🎬", layout="wide")
st.title("✨ Movie & TV Recommender")
st.markdown("""
<style>
.chip {
  display:inline-flex;
  align-items:center;
  background:#f0f2f6;
  border:1px solid #dfe3e8;
  border-radius:999px;
  padding:6px 10px;
  margin:4px 6px 0 0;
  font-size:14px;
  line-height:1.1;
  white-space:nowrap;
  color:#111 !important; /* visible in dark mode */
}
[data-testid="stAppViewContainer"] {
  color-scheme: light dark;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "likes" not in st.session_state:
    st.session_state.likes: List[Dict[str, Any]] = []
if "search_results" not in st.session_state:
    st.session_state.search_results: List[Dict[str, Any]] = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx: Optional[int] = None

# Build/Load index
meta, index_tuple, X = load_index_and_meta()
# Auto-build a small index on first run (fast), then reuse it
AUTO_BUILD_ON_FIRST_RUN = True
DEFAULT_MOVIE_PAGES = 3
DEFAULT_TV_PAGES = 2

meta, index_tuple, X = load_index_and_meta()  # see functions below

if (index_tuple is None or meta.empty) and AUTO_BUILD_ON_FIRST_RUN:
    with st.spinner("First run: building a small corpus & index…"):
        try:
            enriched = quick_build_corpus(pages_movie=DEFAULT_MOVIE_PAGES,
                                          pages_tv=DEFAULT_TV_PAGES)
            if enriched.empty:
                st.warning("Auto-build returned no items. Use the manual builder.")
            else:
                index_tuple, X = build_index_and_save(enriched)  # returns ("faiss", index) or ("sklearn", nn)
                meta, index_tuple, X = load_index_and_meta()
                if index_tuple is not None and not meta.empty:
                    st.success(f"Built index with {len(meta)} titles.")
        except Exception as e:
            st.error(f"Auto-build failed: {e}")

st.divider()

# ---------- Type-ahead search + selection ----------
st.subheader("Pick titles you like")

# A form makes Enter/Return submit, and keeps button inline with the input
with st.form("search_form", clear_on_submit=False):
    search_col, button_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search for a movie or TV show",
            placeholder="e.g., Succession, Oppenheimer, Dune",
            key="search_query",
            label_visibility="visible",   # or "collapsed" if you want to hide the label
        )
    with button_col:
        submitted = st.form_submit_button("Search", use_container_width=True)

# Run the search only when the form is submitted (button click OR Enter)
if submitted and query.strip():
    try:
        j = get_tmdb("/search/multi", {"query": query.strip(), "include_adult": False, "page": 1})
        results = [
            {
                "title": item.get("title") or item.get("name"),
                "year": (item.get("release_date") or item.get("first_air_date") or "????")[:4],
                "media_type": item.get("media_type"),
                "tmdb_id": int(item.get("id")),
                "poster_path": item.get("poster_path"),
            }
            for item in j.get("results", [])
            if item.get("media_type") in {"movie", "tv"}
        ]
        st.session_state.search_results = results
        st.session_state.selected_idx = None
    except Exception as e:
        st.session_state.search_results = []
        st.session_state.selected_idx = None
        st.warning(f"No results found. ({e})")

# Dropdown fed from last search
options = list(range(len(st.session_state.get("search_results", []))))
labels = [
    f"{r['title']} ({r['year']}) — {r['media_type'].upper()}"
    for r in st.session_state.get("search_results", [])
]
selected = st.selectbox(
    "Pick from results",
    options=options if options else [],
    format_func=(lambda i: labels[i] if options else ""),
    index=0 if options else None,
    placeholder="Nothing yet — search above",
)

add_cols = st.columns([1, 6])
with add_cols[0]:
    if st.button("Add selected", disabled=not options):
        r = st.session_state.search_results[selected]
        key_set = {(it["media_type"], it["tmdb_id"]) for it in st.session_state.likes}
        if (r["media_type"], r["tmdb_id"]) not in key_set:
            st.session_state.likes.append(r)

# ---------- Chips Row (single renderer; no duplicates) ----------
st.write("")  # tiny spacer
if st.session_state.likes:
    for it in st.session_state.likes.copy():
        c1, c2 = st.columns([0.96, 0.04])   # keep label and ✕ on the same line
        with c1:
            st.markdown(
                f'<span class="chip">{it["title"]} ({it["media_type"].upper()})</span>',
                unsafe_allow_html=True
            )
        with c2:
            if st.button("✕", key=f"rm_{it['media_type']}_{it['tmdb_id']}", help=f"Remove {it['title']}"):
                st.session_state.likes = [
                    x for x in st.session_state.likes
                    if not (x["media_type"] == it["media_type"] and x["tmdb_id"] == it["tmdb_id"])
                ]
                st.rerun()
else:
    st.caption("No titles selected yet.")

# ---------- Recommendation controls ----------
top_k = st.slider("How many recommendations?", 5, 20, 10)
pop_blend = st.slider("Blend with popularity", 0.0, 1.0, 0.2, 0.05, help="0 = pure similarity, 1 = pure popularity")

if st.button("✨ Get Recommendations"): 
    if not (index_tuple is not None and not meta.empty):
        st.error("No index available. Build or load a corpus first.")
    elif not st.session_state.likes:
        st.warning("Add at least one title you like.")
    else:
        with st.spinner("Computing recommendations…"):
            recs = recommend_from_selected(st.session_state.likes, meta, index_tuple, X, k=top_k, popularity_blend=pop_blend)

        if recs.empty:
            st.info("No recommendations found. Try different titles.")
        else:
            st.caption(f"Top {len(recs)} recommendations")
            for _, r in recs.iterrows():
                col1, col2 = st.columns([1, 3], vertical_alignment="top")
                # inside your recs render loop, in the col1 block
                with col1:
                    purl = None
                    # 1) try from meta
                    if "poster_path" in r and pd.notna(r["poster_path"]):
                        purl = poster_url(r["poster_path"], size="w342")
                    # 2) fallback: fetch details (cached) to get poster
                    if not purl:
                        try:
                            d = fetch_details_rich(r["media_type"], int(r["tmdb_id"]))  # cached by st.cache_data
                            if d.get("poster_path"):
                                purl = poster_url(d["poster_path"], size="w342")
                        except Exception:
                            purl = None
                    if purl:
                        st.image(purl, use_container_width=True)
                    else:
                        st.caption("No poster")
                with col2:
                    name = f"**{r['title']}** — {r['media_type'].upper()}"
                    date = r.get("release_date") or "—"
                    st.markdown(f"{name}  \n*Release/Air:* {date}")
                    if "score" in recs.columns:
                        st.write(f"Relevance score: {r['score']:.3f}")

                    with st.expander("🔎 View full details"):
                        try:
                            d = fetch_details_rich(r["media_type"], int(r["tmdb_id"]))
                            genres = ", ".join(d.get("genres") or [])
                            cast = ", ".join(d.get("cast") or [])
                            crew = ", ".join(d.get("crew") or [])
                            keywords = ", ".join(d.get("keywords") or [])

                            st.write(d.get("overview") or "—")
                            st.write(f"**Genres:** {genres or '—'}")
                            if keywords:
                                st.write(f"**Keywords:** {keywords}")
                            if cast:
                                st.write(f"**Cast (top 10):** {cast}")
                            if crew:
                                st.write(f"**Crew:** {crew}")
                            st.write(f"**Release/Air date:** {d.get('release_date') or '—'}")
                            if d.get("runtime"):
                                st.write(f"**Runtime:** {d['runtime']} min")
                            if d.get("seasons"):
                                st.write(f"**Seasons:** {d['seasons']}")
                            if d.get("episodes"):
                                st.write(f"**Episodes:** {d['episodes']}")
                            if d.get("vote_average") is not None:
                                st.write(f"**User rating:** {d['vote_average']}")
                            if d.get("popularity") is not None:
                                st.write(f"**Popularity:** {int(d['popularity'])}")
                            if d.get("homepage"):
                                st.write(f"[Homepage]({d['homepage']})")
                            if d.get("status"):
                                st.write(f"**Status:** {d['status']}")
                        except requests.HTTPError as e:
                            st.write(f"Could not fetch details: {e}")
