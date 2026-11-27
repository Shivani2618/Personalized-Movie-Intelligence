# Personalized Movie Intelligence · Design 4

A lavender glassmorphism Streamlit redesign of your movie discovery app. The original sparse TF-IDF matrix and cosine-similarity recommendation method are preserved. This source bundle does not include the large movie dataset/model files.

## 1. Keep your existing data
Copy your existing `data` folder into this project folder. It must contain:
- `combined_movies.csv`
- `tfidf_matrix.npz`

Do not regenerate or replace these files for this redesign.

## 2. Run on Windows
Open PowerShell in this folder and run:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If Python 3.12 is not installed, use `python -m venv .venv` instead.

## 3. Optional Where to Watch setup
The rest of the app works without an API key. To enable live India availability, create `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_tmdb_v3_key"
# or
TMDB_BEARER_TOKEN = "your_tmdb_v4_read_token"
```
Never commit secrets to GitHub. On Streamlit Community Cloud, add the same values in App settings → Secrets.

## 4. GitHub / Streamlit Cloud
Upload `app.py`, `requirements.txt`, `train_model.py`, `prepare_dataset.py`, and the `data` files to your private/public repository as appropriate. Keep large files out of Git if your repository limits require external storage, but ensure the deployed app can access both required data files. Set the Streamlit main file to `app.py`, then redeploy.

## Notes
- The app uses the existing `cosine_similarity(matrix[movie_index], matrix)` recommendation calculation.
- No training or dataset preparation is performed on app startup.
- Watchlist and Recently Explored are session-based and reset when the Streamlit session ends.
