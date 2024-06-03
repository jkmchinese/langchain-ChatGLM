from http.client import BAD_REQUEST, NO_CONTENT, OK
import os
from typing import List
import re
import nltk
from numpy import source
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, FilePath

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (VS_ROOT_PATH, EMBEDDING_DEVICE, UPLOAD_ROOT_PATH,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

from crawl_module.crawl import startCrawl

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class Document(BaseModel):
    source: str = Field(..., description="source")
    link: str = Field(..., description="link")
    score: int = Field(..., description="score")
    content: str = Field(..., description="content")


class SearchResult(BaseModel):
    count: int = Field(..., description="Number of source documents")
    documents: List[Document] = Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "count": 5,
                "documents": [],  # 这里将 source_documents 更改为 documents
            }
        }


def get_vs_path(local_doc_id: str):
    return os.path.join(VS_ROOT_PATH, local_doc_id)


def getLink(md: str):
    pattern = r'(\w+)-(\d+)'
    match = re.search(pattern, md)
    if match:
        project_part = match.group(1)
        number_part = match.group(2)
        if project_part == "vmp":
            return f"http://git.rdapp.com/product/vmp2000/issues/{number_part}"
        elif project_part == "eds":
            return f"http://git.rdapp.com/product/eds9000/issues/{number_part}"
        elif project_part == "kp":
            return f"http://git.rdapp.com/product/knowledge-planet/issues/{number_part}"
        else:
            return ""
    else:
        return ""


async def search(
    knowledge_base_id: str = Body(...,
                                  description="Knowledge Base Name", example="vmp"),
    question: str = Body(..., description="Question", example="分级录音存储"),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return SearchResult(
        )
    else:
        searchDocs = local_doc_qa.search(
            query=question, vs_path=vs_path
        )
        sorted_source_documents = sorted(
            searchDocs, key=lambda x: x.metadata['score'])
        return SearchResult(
            count=len(searchDocs),
            documents=[Document(
                score=doc.metadata['score'],
                content=doc.page_content,
                source=os.path.split(doc.metadata['source'])[-1],
                link=getLink(os.path.split(doc.metadata['source'])[-1]))
                for doc in sorted_source_documents]
        )


async def generate(
        knowledge_base_id: str = Body(...,
                                      description="Knowledge Base Name", example="vmp"),
        question: str = Body(None)):
    filepath = os.path.join(UPLOAD_ROOT_PATH, "gitlab", knowledge_base_id)
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path) or not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="校验数据有问题")
    else:
        # UPLOAD_ROOT_PATH
        local_doc_qa.init_knowledge_vector_store(filepath, vs_path)
        return None


async def crawl(
        updatedAfterTime: str = Body(...,
                                     description="updatedAfterTime", example="2024-05-26"),
        question: str = Body(None)):
    count = startCrawl(updatedAfterTime)
    return count


def api_start(host, port):
    global app
    global local_doc_qa

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.post("/search", response_model=SearchResult)(search)
    app.post("/generate", response_model=None)(generate)
    app.post("/crawl", response_model=int)(crawl)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)
