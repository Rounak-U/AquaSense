# Rainfall Prediction using ANN

## Project Overview
Predict rainfall (mm) in India using historical data (1901–2015) with an Artificial Neural Network (ANN). Includes season, subdivision, and previous year rainfall as features.

## File Structure
- `data/` : CSV datasets
- `notebooks/` : Jupyter notebook with ANN
- `app/` : Streamlit interactive app
- `models/` : Saved trained ANN model
- `utils/` : Optional preprocessing functions
- `requirements.txt` : Python dependencies

## How to Run
### Prerequisites

- Python 3.8 or newer
- Git (optional)
- ~100 MB free disk space for datasets and model files

### Quick start (recommended)

1. Create and activate a virtual environment (recommended):

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

2. Install the app dependencies (the Streamlit app dependencies live in `app/requirements.txt`):

	```bash
	pip install -r app/requirements.txt
	```

	If you prefer, you can install Streamlit globally and the other required packages manually.

### Dataset / Model

- Put the main CSV dataset in the `data/` directory. The project expects the file named exactly:

  - `data/rainfall in india 1901-2015.csv`

- A trained model (if available) should be placed in `models/ann_rainfall_model.h5`. If this file is missing, the app will show a warning but can still run for demonstration.

### Running the Streamlit app

From the project root (`/home/rounak/Rainfall`) you can either run the included launcher or directly run Streamlit:

```bash
# using the provided helper script (recommended)
./launch_app.sh

# or run Streamlit directly
streamlit run app/streamlit_app.py --server.port 8501
```

The launcher script performs a few pre-checks (model and data presence) and opens the app on port 8501.

### Training (notes)

- This repository contains analysis / training notebooks in `notebooks/`. If you want to re-train the ANN, open the notebook and run the cells to preprocess data and train the model. After training, export/save the Keras model to `models/ann_rainfall_model.h5` so the Streamlit app can load it.

### Project structure (summary)

- `data/` — place CSVs here
- `notebooks/` — model development & training (Jupyter)
- `app/` — Streamlit front-end (`app/streamlit_app.py`, `app/requirements.txt`)
- `models/` — trained model artifacts
- `launch_app.sh` — convenience script to launch the app

## Troubleshooting

- Port already in use: if port 8501 is occupied, either stop the occupying process or run Streamlit with a different port: `streamlit run app/streamlit_app.py --server.port 8502`.
- Missing model warning: if `models/ann_rainfall_model.h5` is not present the app will warn and run in demo mode. Train the model in `notebooks/` and save it as `models/ann_rainfall_model.h5`.
- Missing data file: if `data/rainfall in india 1901-2015.csv` is absent, some features may not work. Restore the CSV to `data/`.
- Virtualenv errors: ensure you use a Python 3.8+ interpreter and that the `.venv` is activated when installing/running.

## Development tips

- To iterate quickly on the app UI, run Streamlit with `--server.headless true` on a remote machine and use port forwarding or `ngrok` for external access.
- Add print/logging statements in `app/streamlit_app.py` if a behavior is unclear. Streamlit prints logs to the terminal where it runs.

## Contact / Attribution

If you have questions or want to contribute, open an issue or PR. Include details about your environment (OS, Python version) and a short description of the problem.

---

This README was updated to include setup and run instructions for the Streamlit app and guidance for training and troubleshooting.
