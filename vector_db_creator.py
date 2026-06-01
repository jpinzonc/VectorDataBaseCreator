import os
from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_vector_db(input_folder, output_folder, hf_token=None, chunk_size=1000, chunk_overlap=150, embedding_model="all-MiniLM-L6-v2"):
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    if not os.path.isdir(input_folder):
        return {"success": False, "error": f"Input folder does not exist: {input_folder}"}

    os.makedirs(output_folder, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    converter = DocumentConverter()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    processing_summary = []

    supported_exts = ('.pdf', '.txt', '.rtf')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_exts)]

    if not files:
        return {"success": False, "error": "No supported files found in the input folder. Supported: PDF, TXT, RTF."}

    for filename in sorted(files):
        path = os.path.join(input_folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == '.txt':
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    raw_text = fh.read()
            else:
                result = converter.convert(path)
                raw_text = result.document.export_to_markdown()
            chunks = text_splitter.split_text(raw_text)

            doc_count = 0
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": filename, "chunk_id": i}
                ))
                doc_count += 1

            processing_summary.append({
                "status": "success",
                "file": filename,
                "chunks": doc_count
            })

        except Exception as e:
            processing_summary.append({
                "status": "failed",
                "file": filename,
                "chunks": 0,
                "error": str(e)
            })

    if not all_chunks:
        return {"success": False, "error": "No documents could be processed.", "summary": processing_summary}

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=output_folder
    )

    total_successful = sum(1 for item in processing_summary if item['chunks'] > 0)

    return {
        "success": True,
        "total_files": len(files),
        "total_processed": total_successful,
        "total_vectors": len(all_chunks),
        "output_dir": output_folder,
        "summary": processing_summary
    }


def _cleanup_folder(folder):
    import shutil
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def create_vector_db_stream(input_folder, output_folder, hf_token=None, chunk_size=1000, chunk_overlap=150, embedding_model="all-MiniLM-L6-v2"):
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    if not os.path.isdir(input_folder):
        yield {"type": "error", "error": f"Input folder does not exist: {input_folder}"}
        return

    os.makedirs(output_folder, exist_ok=True)

    yield {"type": "init", "message": "Loading embedding model..."}
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        converter = DocumentConverter()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except GeneratorExit:
        _cleanup_folder(output_folder)
        return

    all_chunks = []
    processing_summary = []

    supported_exts = ('.pdf', '.txt', '.rtf')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_exts)]

    if not files:
        yield {"type": "error", "error": "No supported files found in the input folder. Supported: PDF, TXT, RTF."}
        return

    yield {"type": "progress", "current": 0, "total": len(files), "message": f"Found {len(files)} file(s)"}

    for idx, filename in enumerate(sorted(files)):
        path = os.path.join(input_folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            yield {"type": "file_start", "file": filename, "current": idx + 1, "total": len(files)}
        except GeneratorExit:
            _cleanup_folder(output_folder)
            return

        try:
            if ext == '.txt':
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    raw_text = fh.read()
            else:
                result = converter.convert(path)
                raw_text = result.document.export_to_markdown()

            chunks = text_splitter.split_text(raw_text)

            doc_count = 0
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": filename, "chunk_id": i}
                ))
                doc_count += 1

            processing_summary.append({"status": "success", "file": filename, "chunks": doc_count})
            yield {"type": "file_done", "file": filename, "chunks": doc_count, "status": "success"}

        except GeneratorExit:
            _cleanup_folder(output_folder)
            return
        except Exception as e:
            processing_summary.append({"status": "failed", "file": filename, "chunks": 0, "error": str(e)})
            yield {"type": "file_done", "file": filename, "chunks": 0, "status": "failed", "error": str(e)}

        yield {"type": "progress", "current": idx + 1, "total": len(files)}

    if not all_chunks:
        yield {"type": "error", "error": "No documents could be processed.", "summary": processing_summary}
        return

    yield {"type": "saving", "message": "Writing vectors to database..."}
    try:
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=output_folder
        )
    except GeneratorExit:
        _cleanup_folder(output_folder)
        return

    total_successful = sum(1 for item in processing_summary if item['chunks'] > 0)

    yield {"type": "done", "result": {
        "success": True,
        "total_files": len(files),
        "total_processed": total_successful,
        "total_vectors": len(all_chunks),
        "output_dir": output_folder,
        "summary": processing_summary
    }}
