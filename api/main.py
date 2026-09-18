from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI(title="Gemma 4 E4B LoRA/DoRA API")
logger = logging.getLogger(__name__)

class SolveRequest(BaseModel):
    question: str
    method: str
    rank: int

class SolveResponse(BaseModel):
    reasoning: str
    final_answer: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "model": "google/gemma-4-E4B-it",
        "supported_methods": ["lora", "dora"],
        "supported_ranks": [4, 8, 16, 32]
    }

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    # Stub for the API implementation
    return SolveResponse(
        reasoning="This is a stub reasoning.",
        final_answer="42"
    )
