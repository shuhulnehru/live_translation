# FastAPI Starter Project

## Setup

- **Create virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

- **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the server

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

- **Root endpoint**: `/`
- **Items endpoint**: `/items/{item_id}`

Interactive docs are available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

