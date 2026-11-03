# GenZ glasses — HKUST Library Wi-Fi dashboard

Streamlit Community Cloud deploy package for the AISC1000B GenZ glasses **Library Wi-Fi Predictive Advanced** explorer.

- Interactive filters, Plotly charts, auto-calculated insights, and an sklearn User-Count forecast.
- Data: cleaned HKUST Library Wi-Fi usage (`data/wifi_clean.parquet`).

## Main file for Streamlit Cloud

Set **Main file path** to:

```text
app.py
```

## Local run

From this folder (repo root):

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout

```text
app.py                 # Streamlit entry
requirements.txt       # lower bounds only (Cloud-friendly)
runtime.txt            # python-3.12 hint
metrics.json           # snapshot metrics (insights also computed live in app)
data/wifi_clean.parquet
.streamlit/config.toml
DEPLOY.md              # Cloud deploy steps
```
