# Damper ML Analysis — Streamlit deployment

Quick steps to deploy this repo on Streamlit Community Cloud or run locally.

## Run locally
1. Create and activate a Python 3.11 (or 3.10/3.9) virtualenv.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. In Streamlit Cloud, link the GitHub repo and select the branch to deploy.
3. Streamlit will install from `requirements.txt` and run `app.py` automatically.

## Important notes
- Large model files (`lstm_damper_best.pt`, `pinn_transformer_best.pt`, `improved_v2_best.pt`) are stored in the repo. Streamlit Community Cloud has storage and runtime limits — consider using Git LFS or hosting the artifacts externally (S3 / Hugging Face / other blob storage) and modify `app.py` to download them at startup.
- Streamlit Community Cloud does not provide GPUs by default. If your models require GPU for reasonable latency, use a container host with GPU support (Render, AWS, GCP, etc.).
## Export Results
- After running predictions, both **Single Model Analysis** and **Model Comparison** views include a **"Export Results"** button.
- Download CSV files with actual vs. predicted force + displacement & velocity for further analysis.

## If you want Vercel instead
- Vercel expects static sites or serverless functions. Deploying `app.py` directly to Vercel will return NOT_FOUND. Instead, extract an inference API (serverless `api/`) that calls a hosted model, and deploy a frontend/static site on Vercel that calls the API.

If you want, I can: create a small script to download model artifacts from external storage, add Git LFS config, or scaffold a Vercel `api/` inference example.

## Update workflow
Every time you make changes:
```bash
git add .
git commit -m "your message"
git push origin main
```
Streamlit Cloud auto-detects the push and redeploys in ~30 seconds to 2 minutes.