# live_translation

OpenAI-powered video dubbing: upload a video, translate speech with GPT-4o (scene-aware), and download a dubbed MP4.

## Setup

1. **Create virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure secrets** (copy `secrets.yaml.example` to `secrets.yaml` and add your OpenAI API key):

```bash
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with your openai_api_key
```

## Running the server

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

- Upload a video, pick source/target languages, and get a dubbed MP4 to download.
- Swagger UI: `http://127.0.0.1:8000/docs`
