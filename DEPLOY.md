# Deploy on Streamlit Community Cloud

1. Push this folder as a GitHub repository (or put these files at the repo root). Do not invent analytics; this package reuses the existing dashboard app.
2. Open [share.streamlit.io](https://share.streamlit.io/) (Streamlit Community Cloud) and sign in with GitHub.
3. Click **New app**.
4. Choose your **GitHub repo** and branch (usually `main`).
5. Set **Main file path** to:
   ```text
   app.py
   ```
6. Click **Deploy**.
7. Wait for the build (installs `requirements.txt`, then runs `streamlit run app.py`). Open the app URL when ready.

### Notes

- Data path is `./data/wifi_clean.parquet` relative to repo root (same folder as `app.py`).
- File size of the parquet is ~1.7 MB — fine for GitHub.
- If the app shows “Missing … wifi_clean.parquet”, confirm `data/wifi_clean.parquet` is committed and the Main file path is exactly `app.py` at repo root.
- `runtime.txt` is `python-3.12` (a Streamlit-supported runtime with pyarrow wheels).
- `requirements.txt` no longer caps `pyarrow` at `<20`. On the Python 3.14 image, that cap resolved to `pyarrow==19.0.1`, which has no cp314 wheel, so `uv` tried to build from source and failed (`cmake` missing). Without the cap, install uses a release that publishes wheels. After this is on `main`, **Reboot** the app so Cloud reinstalls dependencies.
