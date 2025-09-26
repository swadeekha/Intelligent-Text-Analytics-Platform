 Quick start

1. Create venv:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows

2. Install:
   pip install -r requirements.txt

3. Run:
   python app.py
   Open http://localhost:8000/static/index.html in browser

Notes:
- First run will download small models — take ~10-60s depending on connection.
- For faster demo, ensure CPU/GPU drivers ready or swap to smaller models in code.
