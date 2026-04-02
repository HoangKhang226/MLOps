#!/bin/bash
# Start FastAPI in the background
uvicorn app:app --host 0.0.0.0 --port 8000 &
# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
