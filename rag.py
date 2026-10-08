from pathlib import Path
from sentence_transformers import SentenceTransformer,CrossEncoder
from llm import chat_with_llm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    HasIdCondition
)
import uuid
import hashlib
import json
import tempfile
from threading import RLock
import logging
import shutil
from pypdf import PdfReader

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge" 
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_data"
STATE_FILE = Path(__file__).resolve().parent / "knowledge_state.json"
UPLOADS_DIR = KNOWLEDGE_DIR / "uploads"
KNOWLEDGE_LOCK = RLock()
logger = logging.getLogger(__name__)


class PDFValidationError(ValueError):
    def __init__(self, message, code="invalid_pdf"):
        super().__init__(message)
        self.code = code


def extract_pdf_pages(path):
    """Extract text with original 1-based page numbers; never run OCR."""
    try:
        if Path(path).stat().st_size == 0:
            raise PDFValidationError("PDF文件为空", "empty_pdf")
        with Path(path).open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise PDFValidationError("当前暂不支持加密/需要密码的PDF", "encrypted_pdf")
            if not reader.pages:
                raise PDFValidationError("PDF没有页面", "empty_pdf")
            pages = []
            image_only_pages = []
            scan_like_pages = []
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                meaningful = sum(char.isalnum() for char in text)
                has_images = len(page.images) > 0
                if has_images and meaningful == 0:
                    image_only_pages.append(number)
                if has_images and meaningful < 20:
                    scan_like_pages.append(number)
                pages.append({"page": number, "text": text if meaningful else ""})
            if image_only_pages or len(scan_like_pages) > len(pages) / 2:
                raise PDFValidationError(
                    "PDF含无文字的图片页或主要是扫描图片；当前暂不支持OCR/扫描PDF", "scanned_pdf"
                )
            if not any(page["text"] for page in pages):
                raise PDFValidationError(
                    "PDF没有可提取文字，可能为空白或非文字型PDF；当前暂不支持OCR/扫描PDF", "no_extractable_text"
                )
            return pages
    except PDFValidationError:
        raise
    except OSError:
        raise
    except Exception as error:
        raise PDFValidationError("PDF损坏或文本解析失败；当前仅支持可直接提取文字的PDF") from error


def get_source_name(file_path):
    return Path(file_path).resolve().relative_to(KNOWLEDGE_DIR.resolve()).as_posix()

def chunk_text(text,chunk_size=200,overlap=50):
    chunks=[]
    step=chunk_size-overlap
    for i in range(0,len(text),step):
        chunk=text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

#单独处理某一个文件
def load_file_documents(file_path, source=None, pdf_pages=None):
    source = source or get_source_name(file_path)
    if Path(source).suffix.lower() == ".pdf":
        pages = pdf_pages if pdf_pages is not None else extract_pdf_pages(file_path)
        documents = []
        for page in pages:
            for chunk in chunk_text(page["text"]):
                documents.append({"text": chunk, "source": source,
                                  "chunk_id": len(documents), "page": page["page"]})
        return documents
    text=file_path.read_text(encoding="utf-8-sig")
    chunks=chunk_text(text)
    documents=[]
    for i,chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "source": source,
            "chunk_id": i
        })
    return documents

def make_point_id(document):
    unique_text = (
        document["source"]
        + ":"
        + str(document["chunk_id"])
        + ":"
        + document["text"]
    )
    if "page" in document:
        unique_text += ":page:" + str(document["page"])
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            unique_text
        )
    )

def get_file_hash(file_path):
    content=file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()

def get_current_state():
    state={}
    for file_path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if file_path.is_file() and not file_path.is_symlink() and file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
            state[get_source_name(file_path)]=get_file_hash(file_path)
    return state

def load_state():
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE,"r",encoding="utf-8")as file:
        return json.load(file)

def save_state(state):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=STATE_FILE.parent,
                                         prefix="knowledge_state-", suffix=".tmp", delete=False) as file:
            temp_path = Path(file.name)
            json.dump(state, file, ensure_ascii=False, indent=2)
        temp_path.replace(STATE_FILE)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

#单独处理某个修改或者新增的文件
def upsert_file(filename, documents=None, prune=True):
    file_path = KNOWLEDGE_DIR / filename

    if documents is None:
        documents = load_file_documents(file_path)
    if not documents:
        delete_source(filename)
        return 0

    texts = [
        document["text"]
        for document in documents
    ]

    vectors = embedding_model.encode(texts).tolist()

    points = []

    for document, vector in zip(documents, vectors):
        point_id = make_point_id(document)

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=document
            )
        )

    client.upsert(
        collection_name="ai_knowledge",
        points=points,
        wait=True
    )
    # Only prune old points after embedding and the new upsert have succeeded.
    if prune:
        delete_source(filename, keep_point_ids=[point.id for point in points])
    return len(points)

def delete_source(source, keep_point_ids=None):
    source_filter = Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source))]
    )
    if keep_point_ids:
        source_filter.must_not = [HasIdCondition(has_id=keep_point_ids)]
    client.delete(
        collection_name="ai_knowledge",
        points_selector=FilterSelector(filter=source_filter),
        wait=True
    )


def ingest_file(file_path, staged_path=None):
    file_path = Path(file_path)
    content_path = Path(staged_path) if staged_path is not None else file_path
    source = get_source_name(file_path)
    if not content_path.is_file() or file_path.suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise ValueError("知识资料必须是存在的 txt/md/pdf 文件")
    if staged_path is not None and file_path.suffix.lower() != ".pdf":
        raise ValueError("暂存入库仅用于PDF")
    with KNOWLEDGE_LOCK:
        state = load_state()
        file_hash = get_file_hash(content_path)
        if state.get(source) == file_hash:
            chunks = client.count(
                collection_name="ai_knowledge", exact=True,
                count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ).count
            if staged_path is not None:
                content_path.replace(file_path)
            return {"source": source, "status": "unchanged", "chunks": chunks}
        if file_path.suffix.lower() == ".pdf":
            return _ingest_pdf(file_path, content_path, source, state, file_hash, staged_path is not None)
        chunks = upsert_file(source)
        status = "updated" if source in state else "indexed"
        state[source] = file_hash
        save_state(state)
        return {"source": source, "status": status, "chunks": chunks}


def _ingest_pdf(file_path, content_path, source, state, file_hash, staged):
    pages = extract_pdf_pages(content_path)
    documents = load_file_documents(content_path, source=source, pdf_pages=pages)
    old_points = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name="ai_knowledge", limit=128, offset=offset, with_payload=True, with_vectors=True,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        )
        old_points.extend(records)
        if offset is None:
            break
    backup = None
    file_committed = False
    try:
        if staged and file_path.exists():
            with tempfile.NamedTemporaryFile(dir=file_path.parent, prefix=".previous-", suffix=".tmp", delete=False) as saved:
                backup = Path(saved.name)
            shutil.copy2(file_path, backup)
        # Reuse the existing embedding/upsert; defer pruning until the staged PDF is committed.
        chunks = upsert_file(source, documents=documents, prune=False)
        if staged:
            content_path.replace(file_path)
            file_committed = True
        delete_source(source, keep_point_ids=[make_point_id(document) for document in documents])
        next_state = dict(state)
        next_state[source] = file_hash
        save_state(next_state)
        return {"source": source, "status": "updated" if source in state else "indexed", "chunks": chunks,
                "pages": len(pages), "text_pages": sum(bool(page["text"]) for page in pages),
                "skipped_blank_pages": [page["page"] for page in pages if not page["text"]]}
    except Exception:
        # Compensate partial Qdrant writes as well as file/state commit failures.
        try:
            if old_points:
                client.upsert(collection_name="ai_knowledge", wait=True,
                              points=[PointStruct(id=point.id, vector=point.vector, payload=point.payload) for point in old_points])
            delete_source(source, keep_point_ids=[point.id for point in old_points])
        finally:
            try:
                if file_committed:
                    if backup is not None:
                        backup.replace(file_path)
                    else:
                        file_path.unlink(missing_ok=True)
            finally:
                save_state(state)
        raise
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)


def list_knowledge_sources():
    with KNOWLEDGE_LOCK:
        return sorted(load_state())


def sync_knowledge():
    with KNOWLEDGE_LOCK:
        _sync_knowledge()


def _sync_knowledge():
    old_state = load_state()
    current_state = get_current_state()

    new_files = current_state.keys() - old_state.keys()

    deleted_files = old_state.keys() - current_state.keys()

    modified_files = {
        filename
        for filename in current_state.keys() & old_state.keys()
        if current_state[filename] != old_state[filename]
    }

    next_state = dict(old_state)
    for filename in sorted(new_files | modified_files):
        print(f"{'新增' if filename in new_files else '修改'}文件：{filename}")
        try:
            if Path(filename).suffix.lower() == ".pdf":
                ingest_file(KNOWLEDGE_DIR / filename)
            else:
                upsert_file(filename)
        except PDFValidationError as error:
            logger.warning("跳过无效PDF %s：%s", filename, error)
            continue
        next_state[filename] = current_state[filename]

    for filename in deleted_files:
        print(f"删除文件：{filename}")
        delete_source(filename)
        next_state.pop(filename, None)

    save_state(next_state)

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

reranker=CrossEncoder(
    "BAAI/bge-reranker-base"
)

client=QdrantClient(
    path=str(QDRANT_PATH),
    force_disable_check_same_thread=True
)

if not client.collection_exists(
    collection_name="ai_knowledge"
):

    client.create_collection(
        collection_name="ai_knowledge",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    save_state({})
sync_knowledge()

# Calibrated on evaluation/retrieval_gate_baseline.json (22 positive / 22 negative).
# This is the existing reranker score scale, not an answerability probability.
RETRIEVAL_SCORE_THRESHOLD = 0.5


def retrieve_with_confidence(question):
    with KNOWLEDGE_LOCK:
        hits = _retrieve(question)
    top_score = hits[0]["score"] if hits else None
    if top_score is None or top_score < RETRIEVAL_SCORE_THRESHOLD:
        return {"hits": [], "retrieval_status": "no_hit", "top_score": top_score}
    return {"hits": hits, "retrieval_status": "hit", "top_score": top_score}


def retrieve(question):
    # Keep list-returning callers and Agent tools compatible; gate all their hits.
    return retrieve_with_confidence(question)["hits"]


def _retrieve(question):
    query_vector=embedding_model.encode(question).tolist()
    results=client.query_points(
        collection_name="ai_knowledge",
        query=query_vector,
        limit=10,
        score_threshold=0.4,
        with_payload=True
    ).points

    candidates=[
        result.payload
        for result in results
        if result.payload is not None
    ]
    if not candidates:
        return []
    
    pairs=[
        [question,document["text"]]
        for document in candidates
    ]
    rerank_scores=reranker.predict(pairs)
    ranked=sorted(
        zip(rerank_scores,candidates),
        key=lambda x :x[0],
        reverse=True
    )
    return [
        {"text": document["text"], "source": document["source"],
         "page": document.get("page"), "score": float(score)}
        for score, document in ranked[:3]
    ]


def format_retrieved_context(documents):
    return "\n".join(
        f"[{document['source']}"
        + (f"，第{document['page']}页" if document.get("page") is not None else "")
        + f"]\n{document['text']}"
        for document in documents
    )


def get_retrieval_sources(documents):
    sources = []
    seen = set()
    for document in documents:
        key = (document["source"], document.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


def ask_with_rag(question, system_prompt, retrieval=None):
    if retrieval is None:
        retrieval = retrieve_with_confidence(question)
    if retrieval["retrieval_status"] == "no_hit":
        return "当前知识库中没有找到足够相关的资料。", "", [], "no_hit"
    documents = retrieval["hits"]
    context = format_retrieved_context(documents)
    sources = get_retrieval_sources(documents)

    rag_prompt = f"""
    {system_prompt}

    下面是从知识库中检索到的相关资料：
    {context}

    请结合知识库资料和你已有的知识回答用户的问题。
    知识库资料应优先作为参考，但可以使用你已有的知识进行补充解释。

    回答要求：
    1. 优先保证知识库中的内容得到正确使用。
    2. 可以使用你已有的知识进行补充解释。
    3. 如果补充内容不是来自知识库，请不要与知识库内容冲突。
    4. 回答尽量清晰、适合初学者理解。
    """

    answer = chat_with_llm(question, rag_prompt)

    return answer, context, sources, "hit"
