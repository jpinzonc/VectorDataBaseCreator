import hashlib
import os
import time
from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _write_vectors(documents, embeddings, output_folder, vector_store='chroma',
                   pinecone_api_key=None, pinecone_index='', pinecone_namespace='',
                   pinecone_cloud='aws', pinecone_region='us-east-1'):
    if vector_store == 'chroma':
        os.makedirs(output_folder, exist_ok=True)
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=output_folder
        )
        return output_folder

    if vector_store != 'pinecone':
        raise ValueError(f"Unsupported vector store: {vector_store}")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not configured.")
    if not pinecone_index:
        raise ValueError("A Pinecone index name is required.")

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=pinecone_api_key)
    dimension = len(embeddings.embed_query("vector dimension probe"))

    if not pc.indexes.exists(pinecone_index):
        pc.indexes.create(
            name=pinecone_index,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
        )
        for _ in range(120):
            description = pc.indexes.describe(pinecone_index)
            status = getattr(description, 'status', None)
            if isinstance(status, dict):
                ready = status.get('ready', False)
            else:
                ready = getattr(status, 'ready', False)
            if ready:
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                f"Pinecone index '{pinecone_index}' was not ready after 120 seconds."
            )
    else:
        description = pc.indexes.describe(pinecone_index)
        existing_dimension = getattr(description, 'dimension', None)
        if existing_dimension is None and isinstance(description, dict):
            existing_dimension = description.get('dimension')
        if existing_dimension and int(existing_dimension) != dimension:
            raise ValueError(
                f"Pinecone index '{pinecone_index}' has dimension "
                f"{existing_dimension}, but '{embeddings.model_name}' produces "
                f"{dimension}-dimensional vectors."
            )

    index = pc.index(pinecone_index)
    ids = [
        hashlib.sha256(
            f"{pinecone_namespace}|{doc.metadata.get('source', '')}|"
            f"{doc.metadata.get('chunk_id', '')}|{doc.page_content}".encode('utf-8')
        ).hexdigest()
        for doc in documents
    ]
    for start in range(0, len(documents), 100):
        document_batch = documents[start:start + 100]
        vector_batch = embeddings.embed_documents(
            [doc.page_content for doc in document_batch]
        )
        records = []
        for vector_id, vector, doc in zip(
            ids[start:start + 100], vector_batch, document_batch
        ):
            metadata = dict(doc.metadata)
            metadata['text'] = doc.page_content
            records.append({
                'id': vector_id,
                'values': vector,
                'metadata': metadata,
            })
        upsert_kwargs = {'vectors': records}
        if pinecone_namespace:
            upsert_kwargs['namespace'] = pinecone_namespace
        index.upsert(**upsert_kwargs)
    location = f"pinecone://{pinecone_index}"
    if pinecone_namespace:
        location += f"/{pinecone_namespace}"
    return location


def create_vector_db(input_folder, output_folder, hf_token=None, chunk_size=1000,
                     chunk_overlap=150, embedding_model="all-MiniLM-L6-v2",
                     vector_store='chroma', pinecone_api_key=None,
                     pinecone_index='', pinecone_namespace='',
                     pinecone_cloud='aws', pinecone_region='us-east-1'):
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    if not os.path.isdir(input_folder):
        return {"success": False, "error": f"Input folder does not exist: {input_folder}"}

    if vector_store == 'chroma':
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

    try:
        output_location = _write_vectors(
            all_chunks, embeddings, output_folder, vector_store,
            pinecone_api_key, pinecone_index, pinecone_namespace,
            pinecone_cloud, pinecone_region
        )
    except Exception as e:
        return {"success": False, "error": str(e), "summary": processing_summary}

    total_successful = sum(1 for item in processing_summary if item['chunks'] > 0)

    return {
        "success": True,
        "total_files": len(files),
        "total_processed": total_successful,
        "total_vectors": len(all_chunks),
        "output_dir": output_location,
        "vector_store": vector_store,
        "summary": processing_summary
    }


def _cleanup_folder(folder):
    import shutil
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def create_vector_db_stream(input_folder, output_folder, hf_token=None, chunk_size=1000,
                            chunk_overlap=150, embedding_model="all-MiniLM-L6-v2",
                            vector_store='chroma', pinecone_api_key=None,
                            pinecone_index='', pinecone_namespace='',
                            pinecone_cloud='aws', pinecone_region='us-east-1'):
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    if not os.path.isdir(input_folder):
        yield {"type": "error", "error": f"Input folder does not exist: {input_folder}"}
        return

    if vector_store == 'chroma':
        os.makedirs(output_folder, exist_ok=True)

    yield {"type": "init", "message": "Loading embedding model..."}
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        converter = DocumentConverter()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except GeneratorExit:
        if vector_store == 'chroma':
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
            if vector_store == 'chroma':
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
            if vector_store == 'chroma':
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
        output_location = _write_vectors(
            all_chunks, embeddings, output_folder, vector_store,
            pinecone_api_key, pinecone_index, pinecone_namespace,
            pinecone_cloud, pinecone_region
        )
    except GeneratorExit:
        if vector_store == 'chroma':
            _cleanup_folder(output_folder)
        return
    except Exception as e:
        yield {"type": "error", "error": str(e), "summary": processing_summary}
        return

    total_successful = sum(1 for item in processing_summary if item['chunks'] > 0)

    yield {"type": "done", "result": {
        "success": True,
        "total_files": len(files),
        "total_processed": total_successful,
        "total_vectors": len(all_chunks),
        "output_dir": output_location,
        "vector_store": vector_store,
        "summary": processing_summary
    }}
