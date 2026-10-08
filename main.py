from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from pathlib import Path
import re
import tempfile

from llm import chat_with_llm
from rag import ask_with_rag, ingest_file, list_knowledge_sources, UPLOADS_DIR, KNOWLEDGE_DIR, KNOWLEDGE_LOCK, PDFValidationError


import logging

from langgraph_agent import run_langgraph_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    level: Literal["beginner", "intermediate"] = "beginner"


class ChatResponse(BaseModel):
    level: str
    answer: str


class RagSource(BaseModel):
    source: str
    page: int | None = None


class RagResponse(BaseModel):
    level: str
    answer: str
    retrieved_context: str
    sources: list[RagSource]
    retrieval_status: Literal["hit", "no_hit"]

class LocalAgentSource(BaseModel):
    type: Literal["local"]
    source: str
    page: int | None = None


class WebAgentSource(BaseModel):
    type: Literal["web"]
    title: str
    url: str


class AgentResponse(BaseModel):
    session_id:str
    answer:str
    sources: list[LocalAgentSource | WebAgentSource]
    retrieval_status: Literal["hit", "no_hit"]
    knowledge_source: Literal["local", "web", "none"]
    

class AgentRequest(BaseModel):
    question:str
    session_id:str


def get_prompt(level):
    if level == "beginner":
        return """
        你是一名计算机学习助手。
        用户目前是初学者。

        回答要求：
        1. 先用简单语言解释概念
        2. 尽量使用生活中的类比
        3. 再给出技术上的准确解释
        4. 不要一次使用太多专业术语
        5. 回答控制在 500 字以内
        6. 专业术语第一次出现时必须解释
        7. 不要延伸太多用户没有询问的知识
        """

    elif level == "intermediate":
        return """
        你是一名计算机学习助手。
        用户目前是中等水平的学习者。

        回答要求：
        1. 用相对专业的语言解释概念
        2. 尽量联系计算机学习的知识去解释概念
        3. 引导用户把当前知识和相关知识串联起来
        4. 可以适当使用专业术语
        5. 回答控制在 800 字以内
        6. 重点解释底层机制和知识之间的联系
        7. 最后给出 2 个建议继续学习的相关知识点
        """


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    system_prompt = get_prompt(request.level)

    try:
        answer = chat_with_llm(
            question=request.question,
            system_prompt=system_prompt
        )
    except Exception as e:
        print("大模型调用失败:", e)
        raise HTTPException(
            status_code=500,
            detail="大模型调用失败"
        )

    return {
        "level": request.level,
        "answer": answer
    }


@app.post("/ask", response_model=RagResponse)
def ask(request: ChatRequest):
    system_prompt = get_prompt(request.level)

    try:
        answer, retrieved_context, sources, retrieval_status = ask_with_rag(
            question=request.question,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print("RAG 调用失败:", e)
        raise HTTPException(
            status_code=500,
            detail="RAG 调用失败"
        )

    return {
        "level": request.level,
        "answer": answer,
        "retrieved_context": retrieved_context,
        "sources": sources,
        "retrieval_status": retrieval_status
    }


@app.get("/health")
def examine_health():
    return {
        "status": "ok"
    }


def validate_upload_filename(filename):
    if not filename or filename != filename.strip() or len(filename) > 150:
        raise HTTPException(status_code=400, detail="文件名不能为空，不能有首尾空白，且不能超过150个字符")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', filename) or filename.endswith("."):
        raise HTTPException(status_code=400, detail="文件名包含非法字符或路径")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
    if filename.split(".")[0].upper() in reserved:
        raise HTTPException(status_code=400, detail="不能使用系统保留文件名")
    if Path(filename).suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail="暂时只支持 .txt、.md 和文字型 .pdf 文件")
    return filename


@app.post("/knowledge/upload")
def upload_knowledge(file: UploadFile = File(...)):
    try:
        filename = validate_upload_filename(file.filename)
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="学习资料不能超过5 MiB")
    is_pdf = Path(filename).suffix.lower() == ".pdf"
    if not is_pdf:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="学习资料必须是 UTF-8 编码的文本")
        if not text.strip() or "\x00" in text:
            raise HTTPException(status_code=400, detail="学习资料不能为空或包含 NUL 字符")

    with KNOWLEDGE_LOCK:
        temp_path = None
        try:
            if not UPLOADS_DIR.resolve().is_relative_to(KNOWLEDGE_DIR.resolve()):
                raise OSError("上传目录不在知识库内")
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = UPLOADS_DIR / filename
            if file_path.is_symlink():
                raise HTTPException(status_code=400, detail="不能覆盖符号链接文件")
            with tempfile.NamedTemporaryFile(dir=UPLOADS_DIR, prefix=".upload-", suffix=".tmp", delete=False) as saved:
                temp_path = Path(saved.name)
                saved.write(content)
            if is_pdf:
                try:
                    return ingest_file(file_path, staged_path=temp_path)
                except PDFValidationError as error:
                    raise HTTPException(status_code=400, detail={"code": error.code, "message": str(error)})
                except Exception:
                    logger.exception("PDF upload indexing failed")
                    raise HTTPException(status_code=500, detail="PDF入库失败，请重试上传")
            temp_path.replace(file_path)
        except OSError:
            logger.exception("Knowledge upload save failed")
            raise HTTPException(status_code=500, detail="文件保存失败")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        try:
            return ingest_file(file_path)
        except Exception:
            logger.exception("Knowledge upload indexing failed")
            raise HTTPException(status_code=500, detail="文件已保存但入库失败，请重试上传")


@app.get("/knowledge/sources")
def knowledge_sources():
    return {"sources": list_knowledge_sources()}

@app.post("/agent",response_model=AgentResponse)
def agent_chat(request:AgentRequest):
    try:
        result=run_langgraph_agent(
            user_input=request.question,
            thread_id=request.session_id
        )
        return{
            "session_id":request.session_id,
            **result
        }
    except Exception:
        logger.exception("Agent API failed")
        raise HTTPException(
            status_code=500,
            detail="Agent 调用失败"
        )
