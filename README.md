# Damper ML Analysis

This project is a Streamlit app. The simplest deployment target is Streamlit Community Cloud.

## Run locally
1. Create and activate a Python 3.11 virtualenv.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the app:
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app and select:
   - Repository: this repo
   - Branch: your deploy branch
   - Main file path: `app.py`
4. Deploy. Streamlit will install `requirements.txt` and launch the app automatically.

## Notes
- `plotly` is required at runtime and is included in `requirements.txt`.
- The repo is about 35 MB, so the checked-in `.pt` and `.pkl` artifacts are small enough for a straightforward first deployment.
- Streamlit Community Cloud runs on CPU. If inference latency becomes a problem, move the app to a container host with more control over compute.
- `.vercel/` is local project-link metadata and is not needed for this deployment.

## Export results
After running predictions, both analysis views expose an export button for CSV downloads.
