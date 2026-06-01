# Vector Database Creator

Web application to convert PDF, TXT, and RTF files into a searchable Chroma vector database using HuggingFace embeddings.

## Project Structure

```
├── app.py                      # Flask web application
├── vector_db_creator.py        # Generalized processing module
├── templates/
│   └── index.html              # Web UI
└── requirements.txt            # Python dependencies
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure HuggingFace token

The app needs a HuggingFace token to download the embedding model. Set it as an environment variable:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

> You can get a free token at https://huggingface.co/settings/tokens

## Usage

### 1. Prepare your data

Place PDF, TXT, or RTF files inside a folder on your machine.

### 2. Run the app

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### 3. Create a database

1. **Input Folder** — Select or type the path to your folder with source files
2. **Output Folder** — Auto-filled; you can change it or browse for another location
3. **Database Name** — Auto-filled from the output folder name; edits are appended to the output path
4. **Advanced Settings** (optional) — Click to expand and adjust:
   - **Chunk Size** — How many characters per chunk (default: 1000)
   - **Overlap** — Characters overlapped between chunks (default: 150)
   - **Embedding Model** — HuggingFace embedding model name (default: `all-MiniLM-L6-v2`)
5. Click **Create Database**

A progress modal shows real-time status per file. You can **Cancel** at any time (partial files are cleaned up).

### 4. Results

After processing, a summary card shows:
- Total files found / processed
- Total vectors created
- Per-file status (success/failed with chunk counts)
