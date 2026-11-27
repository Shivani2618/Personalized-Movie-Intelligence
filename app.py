"""Personalized Movie Intelligence — multilingual content-based recommender."""

from pathlib import Path
from urllib.parse import quote_plus
from html import escape
import re
import os
import difflib
import numpy as np
import pandas as pd
import requests
import urllib3
import streamlit as st
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Personalized Movie Intelligence", page_icon="🎬", layout="wide"
)

st.markdown(
    """
<style>
:root {
  --pmi-ink: #17243a;
  --pmi-muted: #64748b;
  --pmi-blue: #2457d6;
  --pmi-blue-soft: #eaf1ff;
  --pmi-coral: #f07865;
  --pmi-line: #e5eaf2;
  --pmi-surface: #ffffff;
}
.stApp {
  background: #f4f7fb;
  color: var(--pmi-ink);
}
[data-testid="stHeader"] { background: rgba(244,247,251,.92); }
[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--pmi-line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1440px; }
h1, h2, h3, h4, p, label, .stMarkdown { color: var(--pmi-ink); }
.hero {
  position: relative; overflow: hidden;
  padding: 30px 34px; border: 1px solid #dce6fb; border-radius: 22px;
  background: linear-gradient(115deg,#ffffff 0%,#edf3ff 65%,#fff0ec 100%);
  margin-bottom: 24px; box-shadow: 0 10px 28px rgba(33,64,120,.06);
}
.hero:after { content: "✦"; position:absolute; right:34px; top:8px; font-size:100px; line-height:1; color:rgba(36,87,214,.09); }
.hero h1 { font-size: 2.15rem; font-weight: 800; letter-spacing: -.7px; margin: 0; color:#172b4d; }
.hero p { color:#52647e; margin:9px 0 0; font-size:1rem; }
.pill { display:inline-block; margin:14px 7px 0 0; padding:6px 11px; border:1px solid #d7e2f7; border-radius:20px; color:#31528b; background:rgba(255,255,255,.78); font-size:12px; font-weight:600; }
.movie-card { background:#fff; border:1px solid #e4eaf3; border-radius:15px; overflow:hidden; margin-bottom:9px; box-shadow:0 5px 18px rgba(27,48,82,.07); transition:transform .18s ease,box-shadow .18s ease; }
.movie-card:hover { transform:translateY(-3px); box-shadow:0 10px 24px rgba(27,48,82,.12); }
.movie-card img { width:100%; height:255px; object-fit:cover; display:block; }
.movie-card .meta { padding:12px 13px 14px; }
.movie-card .title { font-weight:750; min-height:42px; color:#1c2c46; }
.movie-card .sub { font-size:12px; color:#6b7d98; margin-top:5px; }
.small-note { color:#8290a5; font-size:12px; }
.stButton > button, .stLinkButton > a { border-radius:10px; font-weight:650; border:1px solid #d8e2f1; }
.stButton > button[kind="primary"] { background:var(--pmi-blue); border-color:var(--pmi-blue); color:white; }
.stButton > button:hover, .stLinkButton > a:hover { border-color:var(--pmi-blue); color:var(--pmi-blue); }
.stTextInput input, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] { border-radius:10px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--pmi-line); border-radius:13px; padding:12px 15px; box-shadow:0 3px 12px rgba(27,48,82,.04); }
[data-testid="stTabs"] [role="tab"] { font-weight:650; }
[data-testid="stExpander"] { border:1px solid var(--pmi-line); border-radius:12px; background:#fff; }
hr { border-color:#e5eaf2; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading movie catalog and AI recommendation models...")
def load_assets():
    movies_file = DATA / "combined_movies.csv"
    matrix_file = DATA / "tfidf_matrix.npz"
    weighted_matrix_file = DATA / "tfidf_weighted_matrix.npz"

    if not movies_file.exists() or not matrix_file.exists():
        raise FileNotFoundError(
            "Original model files are missing. Check the data folder."
        )
    if not weighted_matrix_file.exists():
        raise FileNotFoundError(
            "Weighted recommendation model is missing. "
            "Run python train_weighted_model.py first."
        )

    movies = pd.read_csv(movies_file, low_memory=False).fillna("")
    matrix = load_npz(matrix_file).tocsr()
    recommendation_matrix = load_npz(weighted_matrix_file).tocsr()

    if len(movies) != matrix.shape[0]:
        raise ValueError(
            "Movie catalog and original TF-IDF matrix row counts do not match."
        )
    if len(movies) != recommendation_matrix.shape[0]:
        raise ValueError(
            "Movie catalog and weighted recommendation matrix row counts do not match."
        )

    movies["display_label"] = movies.apply(
        lambda r: f"{r['title']} ({r['release_year']}) · {r['original_language']} · ID {r['movie_id']}",
        axis=1,
    )
    return movies, matrix, recommendation_matrix


def trailer_url(title):
    return "https://www.youtube.com/results?search_query=" + quote_plus(
        f"{title} official trailer"
    )


def get_tmdb_credential():
    """Read a TMDB v3 key or v4 read token without hard-coding secrets."""
    key = os.getenv("TMDB_API_KEY", "").strip()
    token = os.getenv("TMDB_BEARER_TOKEN", "").strip()
    try:
        key = key or str(st.secrets.get("TMDB_API_KEY", "")).strip()
        token = token or str(st.secrets.get("TMDB_BEARER_TOKEN", "")).strip()
    except Exception:
        # Streamlit secrets.toml is optional for local runs.
        pass
    return key, token


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _resolve_tmdb_ipv4(host):
    """Resolve TMDB through public DNS-over-HTTPS without changing system DNS."""
    resolvers = (
        ("https://dns.google/resolve", {"name": host, "type": "A"}),
        (
            "https://cloudflare-dns.com/dns-query",
            {"name": host, "type": "A"},
        ),
    )
    for endpoint, params in resolvers:
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers={"accept": "application/dns-json"},
                timeout=2,
            )
            response.raise_for_status()
            payload = response.json()
            addresses = []
            for answer in payload.get("Answer", []):
                if answer.get("type") == 1:
                    address = str(answer.get("data", "")).strip()
                    # Only accept IPv4 literals returned by the DNS resolver.
                    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", address):
                        octets = [int(part) for part in address.split(".")]
                        if all(0 <= part <= 255 for part in octets):
                            addresses.append(address)
            if addresses:
                return list(dict.fromkeys(addresses))
        except (requests.RequestException, ValueError, TypeError):
            continue
    return []


def _tmdb_get(url, headers=None, params=None, timeout=15):
    """GET TMDB normally, then retry using DoH-resolved IPs with TLS SNI intact."""
    try:
        return requests.get(
            url, headers=headers, params=params, timeout=min(timeout, 2)
        )
    except requests.RequestException as direct_error:
        from urllib.parse import urlsplit

        parsed_url = urlsplit(url)
        host = (parsed_url.hostname or "").lower()
        if host != "api.themoviedb.org":
            raise

        ip_addresses = _resolve_tmdb_ipv4(host)
        if not ip_addresses:
            raise direct_error

        prepared = requests.Request(
            "GET", url, headers=headers, params=params
        ).prepare()
        last_error = direct_error

        # Two addresses are enough for this CDN fallback; keep failed attempts short.
        for ip_address in ip_addresses[:2]:
            pool = urllib3.HTTPSConnectionPool(
                host=ip_address,
                port=443,
                timeout=urllib3.Timeout(connect=2, read=2, total=3),
                cert_reqs="CERT_REQUIRED",
                ca_certs=requests.certs.where(),
                assert_hostname=host,
                server_hostname=host,
                maxsize=1,
            )
            try:
                request_headers = dict(prepared.headers or {})
                request_headers["Host"] = host
                raw_response = pool.urlopen(
                    "GET",
                    prepared.path_url,
                    headers=request_headers,
                    retries=False,
                    redirect=False,
                    preload_content=True,
                )

                response = requests.Response()
                response.status_code = raw_response.status
                response.headers = requests.structures.CaseInsensitiveDict(
                    raw_response.headers
                )
                response._content = raw_response.data
                response._content_consumed = True
                response.url = prepared.url
                response.reason = raw_response.reason
                response.encoding = requests.utils.get_encoding_from_headers(
                    response.headers
                )
                response.request = prepared
                return response
            except (urllib3.exceptions.HTTPError, OSError) as exc:
                last_error = exc
            finally:
                pool.close()

        raise requests.ConnectionError(
            f"TMDB direct connection failed and DNS fallback could not connect: {last_error}"
        ) from last_error


@st.cache_data(ttl=86400, show_spinner=False, max_entries=2048)
def tmdb_search_movies(title, year, api_key="", bearer_token=""):
    """Search TMDB with auth fallback and retry without year when needed."""
    title = str(title or "").strip()
    if not title:
        return {"status": "empty_title", "results": []}
    if not api_key and not bearer_token:
        return {"status": "missing_key", "results": []}

    auth_options = []
    # Try the bearer token first, but never let a bad token block a valid v3 key.
    if bearer_token:
        auth_options.append(
            (
                {
                    "accept": "application/json",
                    "Authorization": f"Bearer {bearer_token}",
                },
                {},
            )
        )
    if api_key:
        auth_options.append(({"accept": "application/json"}, {"api_key": api_key}))

    year_text = str(year or "").strip()
    valid_year = year_text.lower() not in {"", "nan", "none", "null"}
    if valid_year:
        year_text = year_text.split(".")[0]

    last_status = "no_results"
    for headers, auth_params in auth_options:
        for use_year in ([True, False] if valid_year else [False]):
            params = {"query": title, "include_adult": "false", **auth_params}
            if use_year:
                params["year"] = year_text
            try:
                response = _tmdb_get(
                    "https://api.themoviedb.org/3/search/movie",
                    headers=headers,
                    params=params,
                    timeout=15,
                )
                if response.status_code == 401:
                    last_status = "unauthorized"
                    break
                response.raise_for_status()
                results = response.json().get("results", [])
                if results:
                    return {"status": "ok", "results": results}
                last_status = "no_results"
            except requests.Timeout:
                last_status = "timeout"
                break
            except requests.RequestException:
                last_status = "network_error"
                break
            except (ValueError, TypeError):
                last_status = "bad_response"
                break
    return {"status": last_status, "results": []}


def _best_tmdb_result(results, title, year):
    """Choose the closest title/year match rather than blindly taking result zero."""
    target_title = re.sub(r"[^a-z0-9]+", "", str(title or "").lower())
    target_year = str(year or "").strip().split(".")[0]
    ranked = []
    for item in results or []:
        candidate_title = re.sub(
            r"[^a-z0-9]+",
            "",
            str(item.get("title") or item.get("original_title") or "").lower(),
        )
        title_score = (
            difflib.SequenceMatcher(None, target_title, candidate_title).ratio()
            if target_title and candidate_title
            else 0
        )
        release_date = str(item.get("release_date") or "")
        same_year = bool(target_year and release_date.startswith(target_year))
        score = title_score + (0.35 if same_year else 0)
        if item.get("id"):
            ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[0][1] if ranked else None


@st.cache_data(ttl=86400, show_spinner=False, max_entries=2048)
def find_tmdb_movie_id(title, year, api_key="", bearer_token=""):
    """Resolve a local catalogue title to its TMDB movie ID; cache version includes DNS fallback."""
    search = tmdb_search_movies(title, year, api_key, bearer_token)
    match = _best_tmdb_result(search.get("results", []), title, year)
    return int(match["id"]) if match and match.get("id") else None


@st.cache_data(ttl=21600, show_spinner=False, max_entries=1024)
def fetch_watch_providers(movie_id, api_key="", bearer_token="", region="IN"):
    """Fetch TMDB's country-specific provider data; cache for six hours."""
    try:
        tmdb_id = int(float(movie_id))
    except (TypeError, ValueError, OverflowError):
        return {"status": "invalid_id"}
    endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers"
    headers = {"accept": "application/json"}
    params = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif api_key:
        params["api_key"] = api_key
    else:
        return {"status": "missing_key"}
    try:
        response = _tmdb_get(endpoint, headers=headers, params=params, timeout=12)
        if response.status_code == 401:
            return {"status": "unauthorized"}
        if response.status_code == 404:
            return {"status": "not_found"}
        response.raise_for_status()
        payload = response.json()
        return {
            "status": "ok",
            "region": payload.get("results", {}).get(region.upper()),
        }
    except requests.Timeout:
        return {"status": "timeout"}
    except requests.RequestException:
        return {"status": "network_error"}
    except (ValueError, TypeError):
        return {"status": "bad_response"}


def render_watch_providers(row):
    """Resolve a catalogue title to TMDB and display India streaming providers."""
    api_key, bearer_token = get_tmdb_credential()
    if not api_key and not bearer_token:
        st.warning(
            "TMDB credentials are not configured. Add TMDB_API_KEY or "
            "TMDB_BEARER_TOKEN to .streamlit/secrets.toml to check streaming availability."
        )
        return

    with st.spinner("Checking streaming availability in India..."):
        search = tmdb_search_movies(
            row.get("title", ""), row.get("release_year", ""), api_key, bearer_token
        )
        match = _best_tmdb_result(
            search.get("results", []),
            row.get("title", ""),
            row.get("release_year", ""),
        )
        if not match:
            search_status = search.get("status")
            if search_status == "unauthorized":
                st.error(
                    "TMDB rejected the credentials. Check your local secrets.toml settings."
                )
            elif search_status in {"timeout", "network_error"}:
                st.warning(
                    "TMDB could not be reached. The app tried its DNS fallback too. "
                    "Please check the network and try again."
                )
            elif search_status == "missing_key":
                st.warning(
                    "TMDB credentials are missing. Configure them in .streamlit/secrets.toml."
                )
            elif search_status == "no_results":
                st.info("This movie could not be matched to a TMDB listing.")
            else:
                st.info("TMDB could not complete the movie search. Please try again.")
            return
        tmdb_id = int(match["id"])
        result = fetch_watch_providers(tmdb_id, api_key, bearer_token, region="IN")

    status = result.get("status")
    if status == "missing_key":
        st.warning(
            "TMDB credentials are missing. Configure them in .streamlit/secrets.toml."
        )
        return
    if status == "unauthorized":
        st.error(
            "TMDB rejected the credentials. Check your local secrets.toml settings."
        )
        return
    if status in {"timeout", "network_error"}:
        st.warning("TMDB could not be reached right now. Please try again later.")
        return
    if status != "ok":
        st.info("Streaming availability could not be retrieved for this movie.")
        return

    region_data = result.get("region") or {}
    provider_groups = [
        ("Stream", region_data.get("flatrate", [])),
        ("Free", region_data.get("free", [])),
        ("With ads", region_data.get("ads", [])),
        ("Rent", region_data.get("rent", [])),
        ("Buy", region_data.get("buy", [])),
    ]
    any_provider = False
    seen = set()
    for label, providers in provider_groups:
        unique = []
        for provider in providers or []:
            name = str(provider.get("provider_name", "")).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                unique.append(provider)
        if not unique:
            continue
        any_provider = True
        st.markdown(f"**{label}**")
        cols = st.columns(min(4, len(unique)))
        for i, provider in enumerate(unique):
            with cols[i % len(cols)]:
                logo = provider.get("logo_path")
                if logo:
                    st.image(
                        "https://image.tmdb.org/t/p/w92" + logo,
                        caption=provider.get("provider_name", "Provider"),
                        width=72,
                    )
                else:
                    st.write(provider.get("provider_name", "Provider"))

    if not any_provider:
        st.info(
            "TMDB currently lists no streaming, rental, or purchase providers for India."
        )
    justwatch_url = region_data.get("link")
    if justwatch_url:
        st.link_button("Open JustWatch availability", justwatch_url)


def find_tmdb_poster(title, year, api_key="", bearer_token=""):
    """Find the best matching TMDB poster, retrying title search without year."""
    search = tmdb_search_movies(title, year, api_key, bearer_token)
    match = _best_tmdb_result(search.get("results", []), title, year)
    if match and match.get("poster_path"):
        return POSTER_BASE + match["poster_path"]

    # If the closest result has no poster, use the first result that does.
    for item in search.get("results", []):
        if item.get("poster_path"):
            return POSTER_BASE + item["poster_path"]
    return ""


def resolved_poster_url(row):
    """Return the catalogue poster URL directly.

    Do not hide posters because of temporary server-side network failures.
    """
    return poster_url(row)


def poster_url(row):
    path = str(row.get("poster_path", "")).strip()
    if not path or path.lower() in {"nan", "none", "null"}:
        return ""
    if path.startswith(("https://", "http://")):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return POSTER_BASE + path


@st.cache_data(ttl=86400, show_spinner=False, max_entries=512)
def poster_is_available(url):
    """Check a poster once per day so broken source paths never show as broken icons."""
    if not url:
        return False
    try:
        response = requests.get(
            url,
            stream=True,
            timeout=3,
            headers={"User-Agent": "PersonalizedMovieIntelligence/1.0"},
        )
        valid = (
            response.status_code == 200
            and "image" in response.headers.get("Content-Type", "").lower()
        )
        response.close()
        return valid
    except requests.RequestException:
        return False


def poster_placeholder(row, width=180, match_text=""):
    """A designed fallback shown when the source poster path is missing or returns 404."""
    title = escape(str(row.get("title", "Movie")))
    year = escape(str(row.get("release_year", "")))
    language = escape(str(row.get("original_language", "")).upper())
    match = (
        f'<div style="font-size:12px;color:#496ba9;margin-top:8px">{escape(match_text)}</div>'
        if match_text
        else ""
    )
    return (
        f'<div style="width:100%;max-width:{int(width)}px;min-height:245px;'
        "box-sizing:border-box;border:1px solid #d8e2f1;border-radius:12px;"
        "background:linear-gradient(155deg,#edf3ff,#ffffff 68%,#fff0ec);"
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        'padding:14px;text-align:center;color:#243653;">'
        '<div style="font-size:42px;margin-bottom:12px">🎬</div>'
        f'<div style="font-weight:700;font-size:16px;line-height:1.35">{title}</div>'
        f'<div style="font-size:12px;color:#64748b;margin-top:7px">{year} · {language}</div>'
        '<div style="font-size:11px;color:#7185aa;margin-top:12px">Poster image unavailable</div>'
        f"{match}</div>"
    )


def render_clickable_poster(row, width=180):
    """Show a valid poster or a styled fallback; either opens the trailer search."""
    image = resolved_poster_url(row)
    title = escape(str(row.get("title", "movie")))
    link = escape(trailer_url(row.get("title", "")), quote=True)
    if image:
        content = (
            f'<img src="{escape(image, quote=True)}" alt="{title} poster" '
            f'style="width:100%;max-width:{int(width)}px;border-radius:10px;display:block;">'
        )
    else:
        content = poster_placeholder(row, width)
    st.markdown(
        f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
        f'title="Search YouTube for {title} official trailer" style="text-decoration:none">'
        f"{content}</a>",
        unsafe_allow_html=True,
    )


def recommendation_poster_card(row):
    """Build a recommendation card without ever leaving a broken-image icon."""
    image = resolved_poster_url(row)
    title = escape(str(row.get("title", "Movie")))
    year = escape(str(row.get("release_year", "")))
    language = escape(str(row.get("original_language", "")))
    match = f'{float(row.get("match_percentage", 0)):.1f}% AI match'
    if image:
        visual = (
            f'<img src="{escape(image, quote=True)}" alt="{title} poster" '
            'style="width:100%;height:255px;object-fit:cover;display:block;">'
        )
    else:
        visual = poster_placeholder(row, width=260, match_text=match)
    link = escape(trailer_url(row.get("title", "")), quote=True)
    return (
        f'<a href="{link}" target="_blank" rel="noopener noreferrer" style="text-decoration:none">'
        f'<div class="movie-card">{visual}<div class="meta">'
        f'<div class="title">{title}</div>'
        f'<div class="sub">{year} · {language}</div>'
        f'<div class="sub">{match}</div></div></div></a>'
    )


GENRE_NAMES = (
    "Science Fiction",
    "TV Movie",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Thriller",
    "War",
    "Western",
)
GENRE_TOKEN_FORMS = {name.lower(): tuple(name.lower().split()) for name in GENRE_NAMES}


def split_genres(value):
    """Parse both comma-separated and space-joined TMDB genre labels."""
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return set()

    if any(sep in raw for sep in (",", "|", ";")):
        parts = re.split(r"[,|;]", raw)
        return {part.strip().lower() for part in parts if part.strip()}

    tokens = re.findall(r"[a-z]+", raw.lower())
    found = set()
    i = 0
    ordered_genres = sorted(
        GENRE_TOKEN_FORMS.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    while i < len(tokens):
        matched = None
        for genre, phrase in ordered_genres:
            if tuple(tokens[i : i + len(phrase)]) == phrase:
                matched = (genre, len(phrase))
                break
        if matched:
            found.add(matched[0])
            i += matched[1]
        else:
            i += 1

    return found


_THEME_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "around",
    "because",
    "before",
    "being",
    "between",
    "could",
    "every",
    "from",
    "have",
    "into",
    "just",
    "more",
    "most",
    "other",
    "over",
    "same",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "until",
    "upon",
    "very",
    "were",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "your",
    "film",
    "films",
    "movie",
    "movies",
    "story",
    "stories",
    "based",
    "life",
    "world",
    "years",
    "year",
    "time",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "one",
    "two",
    "new",
    "first",
    "make",
    "made",
    "find",
    "finds",
    "take",
    "takes",
    "back",
    "must",
    "gets",
    "adventure",
    "drama",
    "science",
    "fiction",
    "action",
    "comedy",
    "thriller",
    "romance",
    "fantasy",
    "mystery",
    "crime",
    "family",
    "documentary",
    "animation",
    "history",
    "horror",
    "music",
    "war",
    "western",
    "earth",
    "human",
    "humanity",
}


def _theme_tokens(value):
    tokens = [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", str(value or ""))
    ]
    # Keep stopwords in place so phrase extraction never joins words that
    # were separated in the original sentence.
    return [token for token in tokens if len(token) > 3]


def safe_keywords(a, b, limit=5):
    """Return useful shared concepts, prioritising adjacent-word phrases."""
    tokens_a = _theme_tokens(a)
    tokens_b = _theme_tokens(b)
    if not tokens_a or not tokens_b:
        return []

    def phrases(tokens):
        result = []
        for size in (2, 3):
            for i in range(len(tokens) - size + 1):
                phrase_tokens = tokens[i : i + size]
                if all(token not in _THEME_STOPWORDS for token in phrase_tokens):
                    result.append(" ".join(phrase_tokens))
        return result

    set_bigrams = set(phrases(tokens_b))
    shared_phrases = []
    covered_words = set()
    seen = set()
    for phrase in phrases(tokens_a):
        if phrase in set_bigrams and phrase not in seen:
            shared_phrases.append(phrase)
            seen.add(phrase)
            covered_words.update(phrase.split())

    set_words_b = set(tokens_b)
    shared_words = []
    for token in tokens_a:
        if (
            token in set_words_b
            and token not in _THEME_STOPWORDS
            and token not in covered_words
            and token not in shared_words
        ):
            shared_words.append(token)

    # Prefer phrases over isolated words so the explanation is more descriptive.
    return (shared_phrases + shared_words)[:limit]


def theme_source(row):
    """Use natural plot text first, then Cinematch keywords as extra context."""
    return " ".join(
        str(row.get(column, "") or "") for column in ("overview", "cinematch_tags")
    )


@st.cache_data(show_spinner=False, max_entries=256)
def get_recommendation_indices(movie_index, top_n):
    """Rank movies by weighted text similarity; reject records without a plot."""
    selected_row = movies.iloc[int(movie_index)]
    selected_overview = str(selected_row.get("overview", "")).strip()
    if not selected_overview:
        return [], []

    cosine_scores = cosine_similarity(
        recommendation_matrix[movie_index], recommendation_matrix
    ).ravel()
    cosine_scores[movie_index] = -1

    n = min(int(top_n), len(cosine_scores) - 1)
    if n <= 0:
        return [], []

    # Candidate records must have a real synopsis; genre-only/title-only matches
    # are not useful enough to appear as AI recommendations.
    valid_overview = movies["overview"].astype(str).str.strip().ne("").to_numpy()
    valid_overview[int(movie_index)] = False
    eligible_ids = np.flatnonzero(valid_overview)
    if eligible_ids.size == 0:
        return [], []

    candidate_count = min(max(n * 30, 500), eligible_ids.size)
    eligible_scores = cosine_scores[eligible_ids]
    if float(np.max(eligible_scores)) <= 0.00001:
        return [], []
    if candidate_count < eligible_ids.size:
        local_ids = np.argpartition(eligible_scores, -candidate_count)[
            -candidate_count:
        ]
        candidate_ids = eligible_ids[local_ids]
    else:
        candidate_ids = eligible_ids

    selected_genres = split_genres(selected_row.get("genres", ""))
    selected_language = str(selected_row.get("original_language", "")).strip().lower()
    ranked = []
    for idx in candidate_ids:
        idx = int(idx)
        candidate = movies.iloc[idx]
        candidate_genres = split_genres(candidate.get("genres", ""))
        union = selected_genres | candidate_genres
        genre_similarity = (
            len(selected_genres & candidate_genres) / len(union) if union else 0.0
        )
        candidate_language = str(candidate.get("original_language", "")).strip().lower()
        same_language = bool(
            selected_language and selected_language == candidate_language
        )
        text_similarity = max(0.0, float(cosine_scores[idx]))
        combined = (
            0.97 * text_similarity
            + 0.02 * genre_similarity
            + 0.01 * float(same_language)
        )
        ranked.append((idx, combined))

    ranked.sort(key=lambda item: item[1], reverse=True)
    chosen = ranked[:n]
    return [idx for idx, _ in chosen], [score for _, score in chosen]


# Design 4: Lavender glassmorphism dashboard with a separate weighted recommendation model.
st.markdown(
    """<style>
:root{--ink:#29234a;--muted:#77718f;--violet:#7254c8;--lav:#f2efff;--line:rgba(137,119,190,.20)}
.stApp{background:radial-gradient(circle at 12% 0%,#eee9ff 0,transparent 32%),linear-gradient(135deg,#faf9ff,#f4f2ff 55%,#f8f7ff);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(250,249,255,.78)}
[data-testid="stSidebar"]{background:rgba(255,255,255,.78);backdrop-filter:blur(18px);border-right:1px solid var(--line)}
.block-container{max-width:1500px;padding-top:1.5rem;padding-bottom:3rem}
h1,h2,h3,h4,p,label,.stMarkdown{color:var(--ink)}
.hero{padding:30px 34px;border:1px solid rgba(255,255,255,.9);border-radius:26px;background:linear-gradient(115deg,rgba(255,255,255,.88),rgba(231,224,255,.78));box-shadow:0 14px 40px rgba(79,60,142,.10);backdrop-filter:blur(18px);margin-bottom:22px}
.hero h1{font-size:2.25rem;font-weight:850;letter-spacing:-.8px;color:#30265c;margin:0}.hero p{color:#746c91;margin:8px 0 0}
.pill{display:inline-block;margin:13px 7px 0 0;padding:6px 12px;border:1px solid #ded6ff;border-radius:20px;color:#6048ad;background:rgba(255,255,255,.7);font-size:12px;font-weight:650}
.stButton>button,.stLinkButton>a{border-radius:12px;border:1px solid #ded7f4;font-weight:650;min-height:2.55rem;background:rgba(255,255,255,.78)}
.stButton>button[kind="primary"]{background:linear-gradient(110deg,#7254c8,#9275e8);border-color:#7254c8;color:#fff}
.stButton>button:hover,.stLinkButton>a:hover{border-color:#8065d1;color:#6248b7}
.stTextInput input,.stSelectbox [data-baseweb="select"],.stMultiSelect [data-baseweb="select"]{border-radius:12px;background:rgba(255,255,255,.82)}
[data-testid="stMetric"]{background:rgba(255,255,255,.78);border:1px solid rgba(255,255,255,.95);border-radius:17px;padding:14px 16px;box-shadow:0 8px 25px rgba(72,56,125,.07);backdrop-filter:blur(14px)}
[data-testid="stTabs"] [role="tab"]{font-weight:650}.movie-card{background:rgba(255,255,255,.86);border:1px solid rgba(255,255,255,.95);border-radius:17px;overflow:hidden;margin-bottom:10px;box-shadow:0 8px 25px rgba(72,56,125,.09);transition:transform .18s ease}.movie-card:hover{transform:translateY(-3px)}
.movie-card img{width:100%;height:255px;object-fit:cover;display:block}.movie-card .meta{padding:12px 13px 14px}.movie-card .title{font-weight:750;min-height:42px;color:#30265c}.movie-card .sub{font-size:12px;color:#80799a;margin-top:5px}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:15px;background:rgba(255,255,255,.68)}
hr{border-color:var(--line)}

.filter-panel{padding:16px 18px 10px;border:1px solid rgba(255,255,255,.92);border-radius:20px;background:rgba(255,255,255,.52);box-shadow:0 10px 28px rgba(87,68,150,.06);backdrop-filter:blur(16px);margin:8px 0 20px}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px!important;border-color:rgba(255,255,255,.96)!important;background:rgba(255,255,255,.72)!important;box-shadow:0 10px 28px rgba(87,68,150,.07)!important}
[data-testid="stImage"] img{border-radius:12px;object-fit:cover}
.mood-tile{padding:18px;border:1px solid #e4dcff;border-radius:18px;background:linear-gradient(130deg,rgba(255,255,255,.94),rgba(235,229,255,.78));box-shadow:0 8px 24px rgba(87,68,150,.07);font-weight:750;color:#4a397f}
.mood-tile small{display:block;color:#81789e;font-weight:500;margin-top:6px}
</style>""",
    unsafe_allow_html=True,
)

# Premium cinematic polish: brighter lavender glass, stronger hierarchy, and cleaner cards.
st.markdown(
    """<style>
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(199,181,255,.48),transparent 34%),radial-gradient(ellipse at 95% 12%,rgba(255,211,224,.42),transparent 30%),linear-gradient(145deg,#fbfaff 0%,#f1edff 52%,#fff8fc 100%)}
[data-testid="stHeader"]{background:rgba(250,248,255,.76);backdrop-filter:blur(16px)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(255,255,255,.94),rgba(245,240,255,.92));border-right:1px solid #e6ddfb;box-shadow:8px 0 30px rgba(79,55,145,.045)}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:8px 11px;border-radius:12px;margin:2px 0;transition:all .18s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#eee8ff}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"]{background:linear-gradient(100deg,#eee8ff,#f8edff);border:1px solid #dfd2ff}
.block-container{max-width:1540px;padding-top:1.15rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:34px 38px;border:1px solid rgba(255,255,255,.95);border-radius:28px;background:linear-gradient(112deg,rgba(255,255,255,.96),rgba(235,228,255,.92) 58%,rgba(255,237,245,.9));box-shadow:0 18px 48px rgba(91,67,157,.12);margin-bottom:24px}
.hero:after{content:'✦';position:absolute;right:35px;top:-24px;font-size:150px;color:rgba(126,91,204,.09);pointer-events:none}
.hero h1{font-size:2.55rem;line-height:1.12;font-weight:900;letter-spacing:-1.2px;background:linear-gradient(95deg,#34246c,#7652c8 58%,#bd659a);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{font-size:1.04rem;color:#71668e;max-width:760px}
.pill{border-color:#ded3fb;color:#6046a8;background:rgba(255,255,255,.84);box-shadow:0 3px 10px rgba(91,67,157,.05)}
.filter-panel{padding:18px 20px 12px;border:1px solid rgba(255,255,255,.96);border-radius:22px;background:rgba(255,255,255,.72);box-shadow:0 12px 34px rgba(91,67,157,.075);backdrop-filter:blur(18px)}
.stTextInput input,.stSelectbox [data-baseweb="select"],.stMultiSelect [data-baseweb="select"]{border-radius:13px!important;border-color:#ded5f3!important;background:rgba(255,255,255,.95)!important;min-height:2.8rem}
.stButton>button,.stLinkButton>a{border-radius:13px!important;min-height:2.7rem;font-weight:700;transition:all .18s ease;box-shadow:0 4px 12px rgba(93,67,158,.055)}
.stButton>button[kind="primary"]{background:linear-gradient(105deg,#6746bd,#9470e8)!important;border:0!important;color:white!important;box-shadow:0 8px 18px rgba(111,78,196,.22)}
.stButton>button:hover,.stLinkButton>a:hover{transform:translateY(-1px);box-shadow:0 8px 18px rgba(93,67,158,.12)}
[data-testid="stMetric"],[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid rgba(255,255,255,.98)!important;border-radius:19px!important;background:rgba(255,255,255,.83)!important;box-shadow:0 10px 30px rgba(91,67,157,.075)!important}
[data-testid="stImage"] img{border-radius:14px;box-shadow:0 9px 22px rgba(43,32,83,.13)}
.movie-card{border:1px solid rgba(255,255,255,.98);border-radius:19px;box-shadow:0 10px 30px rgba(73,54,127,.10)}
.movie-card img{height:285px;object-fit:cover}
.mood-tile{border:1px solid #e2d8ff;background:linear-gradient(135deg,#fff, #eee8ff 72%,#fff0f6);box-shadow:0 10px 25px rgba(91,67,157,.08)}
[data-testid="stTabs"] [role="tab"]{padding:10px 16px;border-radius:12px 12px 0 0}
[data-testid="stExpander"]{border-radius:16px!important;border-color:#e5dcf7!important;background:rgba(255,255,255,.78)!important}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<style>
.cinema-hero{position:relative;display:flex;align-items:center;justify-content:space-between;gap:28px;min-height:310px;padding:38px 34px;border-radius:28px;overflow:hidden;margin:4px 0 28px;background:linear-gradient(100deg,rgba(25,15,54,.97) 0%,rgba(42,25,83,.93) 47%,rgba(55,35,91,.52) 100%);box-shadow:0 22px 55px rgba(48,28,91,.22)}
.cinema-copy{position:relative;z-index:2;max-width:55%}.cinema-copy .eyebrow{color:#d8c8ff;font-size:11px;font-weight:800;letter-spacing:2px}.cinema-copy h1{color:#fff!important;font-size:clamp(2rem,3.5vw,3.25rem);font-weight:900;line-height:1.05;margin:12px 0}.cinema-copy p{color:#e4dcf7!important;font-size:1.05rem;max-width:550px}.hero-tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.hero-tags span{padding:7px 11px;border-radius:30px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.2);color:white;font-size:11px}.hero-posters{display:flex;align-items:center;gap:9px;transform:rotate(3deg);margin-right:-18px}.hero-posters img{width:clamp(65px,8vw,112px);height:180px;object-fit:cover;border-radius:12px;box-shadow:0 12px 25px rgba(0,0,0,.32);border:1px solid rgba(255,255,255,.25)}
@media(max-width:800px){.cinema-hero{display:block;padding:25px}.cinema-copy{max-width:100%}.hero-posters{margin:22px 0 0;transform:none}.hero-posters img{height:125px}}
</style>""",
    unsafe_allow_html=True,
)

try:
    movies, matrix, recommendation_matrix = load_assets()
except Exception as exc:
    st.error(f"Could not load the project model: {exc}")
    st.info(
        "Check that data/combined_movies.csv and data/tfidf_matrix.npz are inside this project folder."
    )
    st.stop()

# Normalize display values only. Source data and model files are never modified.
for col in [
    "title",
    "overview",
    "genres",
    "original_language",
    "tags",
    "cinematch_tags",
    "display_label",
    "movie_id",
    "release_year",
    "rating",
    "popularity",
]:
    if col not in movies.columns:
        movies[col] = ""
movies["_rating"] = pd.to_numeric(movies["rating"], errors="coerce").fillna(0)
movies["_popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0)
movies["_year"] = pd.to_numeric(movies["release_year"], errors="coerce")
language_codes = sorted(
    x for x in movies.original_language.astype(str).unique() if x and x.lower() != "nan"
)
genres = sorted(
    {
        {"science fiction": "Science Fiction", "tv movie": "TV Movie"}.get(
            genre, genre.title()
        )
        for cell in movies.genres.astype(str)
        for genre in split_genres(cell)
    }
)

for key, default in [
    ("watchlist", []),
    ("recently_explored", []),
    ("selected_label", ""),
    ("last_recommendations", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Restore the sidebar navigation variable before any page-specific rendering.
NAV_PAGES = [
    "Home",
    "Explore Movies",
    "Browse by Genre",
    "Trending",
    "MoodFlix",
    "AI Recommendations",
    "My Watchlist",
    "Recently Explored",
    "Compare Movies",
    "Where to Watch",
    "Settings",
]
if "sidebar_page" not in st.session_state:
    st.session_state.sidebar_page = "Home"
page = st.sidebar.radio("Navigate", NAV_PAGES, key="sidebar_page")


def remember_movie(row):
    item = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    mid = str(item.get("movie_id", ""))
    history = [
        x
        for x in st.session_state.recently_explored
        if str(x.get("movie_id", "")) != mid
    ]
    history.insert(0, item)
    st.session_state.recently_explored = history[:15]


def save_movie(row, key_suffix=""):
    item = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    if str(item.get("movie_id")) not in {
        str(x.get("movie_id")) for x in st.session_state.watchlist
    }:
        st.session_state.watchlist.append(item)
        st.toast("Added to your watchlist")


def movie_card(row, key_prefix="movie"):
    """Poster-first catalogue card with useful movie actions."""
    image = resolved_poster_url(row)
    with st.container(border=True):
        if image:
            st.image(image, use_container_width=True)
        else:
            st.markdown(poster_placeholder(row, width=320), unsafe_allow_html=True)
        st.markdown(f"**{row.get('title','Movie')} ({row.get('release_year','')})**")
        st.caption(
            f"{row.get('genres') or 'Genre not listed'} · {str(row.get('original_language','')).upper()} · ★ {row.get('rating') or 'N/A'}"
        )
        overview = str(row.get("overview", ""))
        st.write(
            (overview[:125] + "…")
            if len(overview) > 125
            else (overview or "No synopsis available.")
        )
        a, b = st.columns(2)
        with a:
            st.link_button(
                "▶ Trailer", trailer_url(row.get("title", "")), use_container_width=True
            )
        with b:
            if str(row.get("movie_id")) not in {
                str(x.get("movie_id")) for x in st.session_state.watchlist
            }:
                if st.button(
                    "＋ Save",
                    key=f"{key_prefix}_{row.get('movie_id')}",
                    use_container_width=True,
                ):
                    save_movie(row)
                    st.rerun()
            else:
                st.button(
                    "✓ Saved",
                    key=f"{key_prefix}_saved_{row.get('movie_id')}",
                    disabled=True,
                    use_container_width=True,
                )


def render_grid(frame, key_prefix="grid", limit=20):
    if frame.empty:
        st.info("No movies match these filters. Try changing your search or filters.")
        return
    shown = frame.head(limit)
    for start in range(0, len(shown), 4):
        cols = st.columns(4, gap="medium")
        for col, (_, row) in zip(cols, shown.iloc[start : start + 4].iterrows()):
            with col:
                movie_card(row, f"{key_prefix}_{start}")
    if len(frame) > limit:
        st.caption(
            f"Showing {limit} of {len(frame):,} matching movies. Use filters to narrow the catalogue."
        )


# Cinematic hero with a poster strip inspired by streaming-service home screens.
hero_pool = movies.sort_values(["_popularity", "_rating"], ascending=False).head(6)
hero_imgs = []
for _, _r in hero_pool.iterrows():
    _u = resolved_poster_url(_r)
    if _u:
        hero_imgs.append(_u)
hero_strip = "".join(f'<img src="{escape(u, quote=True)}" />' for u in hero_imgs[:6])
st.markdown(
    f"""<div class="cinema-hero"><div class="cinema-copy"><div class="eyebrow">YOUR PERSONAL CINEMA</div><h1>Personalized Movie Intelligence</h1><p>Find your next favourite. Discover films picked around the stories you love.</p><div class="hero-tags"><span>AI-powered discovery</span><span>Multilingual movies</span><span>Personal watchlist</span></div></div><div class="hero-posters">{hero_strip}</div></div>""",
    unsafe_allow_html=True,
)

# The old catalogue search bar has been removed. Movie selection is now the first action on Home.
base = movies.copy()

if page == "Home":
    st.markdown("### Choose a movie to get personal recommendations")
    st.caption(
        "Start with a movie you love. The AI will find similar titles from your existing catalogue."
    )
    home_options = (
        movies.sort_values("_popularity", ascending=False)
        .head(15000)
        .display_label.tolist()
    )
    home_choice = st.selectbox(
        "Choose a movie",
        home_options,
        key="home_movie_picker",
        label_visibility="visible",
    )

    def open_recommendations_for_home_movie(movie_label):
        st.session_state.selected_label = movie_label
        st.session_state.auto_generate_recommendations = True
        st.session_state.sidebar_page = "AI Recommendations"

    st.button(
        "✨ Get my recommendations",
        type="primary",
        use_container_width=True,
        on_click=open_recommendations_for_home_movie,
        args=(home_choice,),
    )
    st.markdown("#### ✨ Featured from your movie universe")
    # A visual poster shelf near the top makes Home feel like a streaming discovery page.
    featured_pool = movies.sort_values(
        ["_popularity", "_rating"], ascending=False
    ).head(80)
    featured_rows = []
    for _, featured_row in featured_pool.iterrows():
        if len(featured_rows) >= 5:
            break
        featured_rows.append(featured_row)
    poster_cols = st.columns(5, gap="medium")
    for poster_col, (_, featured_row) in zip(poster_cols, enumerate(featured_rows)):
        with poster_col:
            featured_poster = resolved_poster_url(featured_row)
            if featured_poster:
                st.image(featured_poster, use_container_width=True)
            else:
                st.markdown(
                    poster_placeholder(featured_row, width=240), unsafe_allow_html=True
                )
            st.markdown(f"**{escape(str(featured_row.get('title','Movie')))}**")
            st.caption(
                f"{featured_row.get('release_year','')} · ★ {featured_row.get('rating') or 'N/A'}"
            )
    a, b, c = st.columns(3)
    a.metric("Movies in your library", f"{len(movies):,}")
    b.metric("Languages", f"{len(language_codes):,}")
    c.metric("Saved for later", len(st.session_state.watchlist))
    st.markdown("### ✨ Popular picks for you")
    render_grid(base.sort_values("_popularity", ascending=False), "home", 16)
    st.markdown("### Explore by mood")
    mood_cols = st.columns(3)
    mood_links = [
        ("🍿 Feel-good", "Comedy"),
        ("🕵️ Mystery night", "Mystery"),
        ("🚀 Fantasy worlds", "Fantasy"),
    ]
    for col, (label, genre_name) in zip(mood_cols, mood_links):
        with col:
            st.markdown(
                f"<div class='mood-tile'><span>{label}</span><small>Discover {genre_name} films</small></div>",
                unsafe_allow_html=True,
            )
    st.caption(
        f"{len(base):,} titles currently match your search and filters. Choose a section from the sidebar to explore more."
    )

elif page == "Explore Movies":
    st.markdown("### Explore the full movie library")
    sort_by = st.selectbox(
        "Sort results",
        ["Most popular", "Highest rated", "Newest first", "Oldest first"],
        key="explore_sort",
    )
    pool = base.copy()
    if sort_by == "Most popular":
        pool = pool.sort_values("_popularity", ascending=False)
    elif sort_by == "Highest rated":
        pool = pool.sort_values(["_rating", "_popularity"], ascending=False)
    elif sort_by == "Newest first":
        pool = pool.sort_values("_year", ascending=False, na_position="last")
    else:
        pool = pool.sort_values("_year", ascending=True, na_position="last")
    st.caption(f"{len(pool):,} movies found using the filters above")
    render_grid(pool, "explore", 32)

elif page == "Browse by Genre":
    st.markdown("### Find your favourite genre")
    chosen = st.selectbox("Choose a genre", genres, key="genre_page")
    pool = base[
        base.genres.astype(str).str.contains(re.escape(chosen), case=False, na=False)
    ].sort_values("_popularity", ascending=False)
    st.caption(f"{len(pool):,} titles tagged with {chosen}")
    render_grid(pool, "genre", 24)

elif page == "Trending":
    st.markdown("### Trending from your movie dataset")
    st.caption("Sorted by the popularity values already present in your catalogue.")
    render_grid(base.sort_values("_popularity", ascending=False), "trend", 24)

elif page == "MoodFlix":
    st.markdown("### What are you in the mood for?")
    profiles = {
        "Feel-good & funny": ["Comedy", "Family", "Animation"],
        "Thrill & suspense": ["Thriller", "Mystery", "Crime", "Action"],
        "Love & romance": ["Romance", "Comedy"],
        "Fantasy & adventure": ["Fantasy", "Adventure", "Science Fiction"],
        "Emotional stories": ["Drama", "Romance", "History"],
        "Curious & thoughtful": ["Documentary", "History", "Mystery"],
    }
    mood = st.selectbox("Choose a mood", list(profiles))
    language = st.selectbox("Language", ["Any"] + language_codes, key="mood_language")
    rating = st.slider("Minimum rating", 0.0, 10.0, 5.0, 0.5, key="mood_rating")
    pattern = "|".join(re.escape(x) for x in profiles[mood])
    pool = base[
        base.genres.astype(str).str.contains(pattern, case=False, na=False)
    ].copy()
    if language != "Any":
        pool = pool[pool.original_language == language]
    pool = pool[pool._rating >= rating].sort_values(
        ["_rating", "_popularity"], ascending=False
    )
    render_grid(pool, "mood", 12)

elif page == "AI Recommendations":
    st.markdown("### Discover movies that feel like your favourites")
    options = (
        movies.sort_values("_popularity", ascending=False)
        .head(10000)
        .display_label.tolist()
    )
    if st.session_state.selected_label not in options:
        st.session_state.selected_label = options[0] if options else ""
    selected_label = st.selectbox(
        "Choose a movie",
        options,
        index=(
            options.index(st.session_state.selected_label)
            if st.session_state.selected_label in options
            else 0
        ),
    )
    previous_label = st.session_state.get("active_recommendation_label", "")
    if previous_label != selected_label:
        st.session_state.last_recommendations = []
        st.session_state.active_recommendation_label = selected_label
    st.session_state.selected_label = selected_label
    selected_index = int(movies.index[movies.display_label == selected_label][0])
    selected = movies.loc[selected_index]
    remember_movie(selected)
    with st.expander("Selected movie details", expanded=True):
        left, right = st.columns([1, 4])
        with left:
            render_clickable_poster(selected, 170)
        with right:
            st.subheader(f"{selected.title} ({selected.release_year})")
            st.caption(
                f"{selected.genres or 'Genres not listed'} · {selected.original_language} · ★ {selected.rating or 'N/A'}"
            )
            st.write(selected.overview or "No synopsis available.")
            st.link_button("▶ Official trailer search", trailer_url(selected.title))
    n = st.slider("Number of recommendations", 4, 12, 8)
    generate_now = st.button(
        "✨ Find similar movies", type="primary", use_container_width=True
    )
    auto_now = bool(st.session_state.pop("auto_generate_recommendations", False))
    if generate_now or auto_now:
        with st.spinner("Finding movies that match your selection..."):
            ids, scores = get_recommendation_indices(selected_index, n)
            recs = []
            for idx, score in zip(ids, scores):
                if idx == selected_index:
                    continue
                row = movies.iloc[idx].to_dict()
                row["match_percentage"] = round(
                    min(0.99, max(0.0, float(score))) * 100, 1
                )
                row["shared_keywords"] = safe_keywords(
                    theme_source(selected),
                    theme_source(row),
                )
                recs.append(row)
        st.session_state.last_recommendations = recs
    recs = st.session_state.last_recommendations
    if recs:
        st.markdown(f"#### Because you selected {selected.title}")
        for i in range(0, len(recs), 2):
            cols = st.columns(2)
            for col, item in zip(cols, recs[i : i + 2]):
                with col:
                    st.markdown(
                        recommendation_poster_card(item), unsafe_allow_html=True
                    )
                    st.caption(
                        "Similarity score: " + str(item["match_percentage"]) + "%"
                    )
                    if item.get("shared_keywords"):
                        st.caption(
                            "Shared themes: " + ", ".join(item["shared_keywords"])
                        )
                    if st.button(
                        "＋ Add to watchlist", key=f"ai_save_{item['movie_id']}"
                    ):
                        save_movie(item)
                        st.rerun()
                    with st.expander("Why this match?"):
                        st.write(item.get("overview") or "No synopsis available.")
                        st.caption(
                            "Match score combines weighted title, plot, genre and Cinematch metadata similarity, with small genre and language tie-breakers."
                        )
    else:
        st.info(
            "Choose a movie and click Find similar movies to generate recommendations."
        )

elif page == "My Watchlist":
    st.markdown("### Your saved movie collection")
    if st.session_state.watchlist:
        if st.button("Clear watchlist"):
            st.session_state.watchlist = []
            st.rerun()
        render_grid(pd.DataFrame(st.session_state.watchlist), "saved", 30)
    else:
        st.info(
            "Your watchlist is empty. Save films from Explore or AI Recommendations."
        )

elif page == "Recently Explored":
    st.markdown("### Recently explored")
    if st.session_state.recently_explored:
        render_grid(pd.DataFrame(st.session_state.recently_explored), "recent", 15)
    else:
        st.info(
            "Movies you open in AI Recommendations will appear here during this session."
        )

elif page == "Compare Movies":
    st.markdown("### Compare two movies")
    opts = (
        movies.sort_values("_popularity", ascending=False)
        .head(10000)
        .display_label.tolist()
    )
    x, y = st.columns(2)
    with x:
        la = st.selectbox("Movie A", opts, key="cmp_a")
    with y:
        lb = st.selectbox("Movie B", opts, index=min(1, len(opts) - 1), key="cmp_b")
    ia = int(movies.index[movies.display_label == la][0])
    ib = int(movies.index[movies.display_label == lb][0])
    if ia == ib:
        st.info("Select two different movies.")
    else:
        similarity = float(cosine_similarity(matrix[ia], matrix[ib])[0, 0])
        st.metric("Thematic similarity", f"{similarity*100:.1f}%")
        common = safe_keywords(movies.iloc[ia].tags, movies.iloc[ib].tags, 8)
        if common:
            st.write("Shared themes: " + ", ".join(common))
        render_grid(movies.iloc[[ia, ib]], "compare", 2)

elif page == "Where to Watch":
    st.markdown("### Find where a movie is streaming")
    st.caption(
        "Availability is checked for India. Streaming catalogues change often, so verify on the provider before watching."
    )
    opts = (
        movies.sort_values("_popularity", ascending=False)
        .head(20000)
        .display_label.tolist()
    )
    chosen = st.selectbox("Choose a movie", opts, key="watch_provider_movie")
    row = movies[movies.display_label == chosen].iloc[0]
    left, right = st.columns([1, 2.2])
    with left:
        poster = resolved_poster_url(row)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown(poster_placeholder(row, width=280), unsafe_allow_html=True)
    with right:
        st.subheader(f"{row.title} ({row.release_year})")
        st.caption(
            f"{row.genres or 'Genre not listed'} · {str(row.original_language).upper()} · ★ {row.rating or 'N/A'}"
        )
        render_watch_providers(row)
        st.caption(
            "Only platforms returned by TMDB for India are listed above. If no providers are returned, TMDB currently has no India availability record for this title. The availability link, when supplied by TMDB, opens the JustWatch listing."
        )

elif page == "Settings":
    st.markdown("### Project settings & model information")
    st.write(
        "The interface reads the movie catalogue, original TF-IDF matrix, and separate weighted recommendation matrix from the local data/ folder."
    )
    st.metric("Catalogue rows", f"{len(movies):,}")
    st.metric(
        "Original TF-IDF matrix shape", f"{matrix.shape[0]:,} × {matrix.shape[1]:,}"
    )
    st.metric(
        "Weighted recommendation matrix shape",
        f"{recommendation_matrix.shape[0]:,} × {recommendation_matrix.shape[1]:,}",
    )
    if "poster_path" in movies.columns:
        poster_count = (
            movies["poster_path"]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": "", "null": ""})
            .ne("")
            .sum()
        )
        st.metric(
            "Movies with poster paths in dataset", f"{poster_count:,} / {len(movies):,}"
        )
        if poster_count == 0:
            st.error(
                "Your local combined_movies.csv has no poster_path values. Posters cannot appear until poster paths are added to the dataset or fetched from a working image API."
            )
        else:
            st.success(
                "Poster paths are present in your dataset. The app now displays them directly without a server-side availability check."
            )
    else:
        st.error(
            "The combined_movies.csv file does not contain a poster_path column. The dataset needs poster metadata added."
        )
    st.caption(
        "AI Recommendations use the separate weighted matrix. Compare Movies continues to use the original TF-IDF matrix. Neither model nor the dataset is rebuilt by this app."
    )
    st.markdown(
        "**Optional TMDB credentials**: set TMDB_API_KEY or TMDB_BEARER_TOKEN in Streamlit secrets to enable India provider lookups. No key is required for the rest of the app."
    )

st.divider()
st.markdown(
    '<p style="text-align:center;color:#8982a5;font-size:12px">Personalized Movie Intelligence · Existing TF-IDF + Cosine Similarity engine · Streamlit</p><p style="text-align:center;color:#8982a5;font-size:11px">Where-to-watch availability powered by JustWatch via TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</p>',
    unsafe_allow_html=True,
)
