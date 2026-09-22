from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from laya import Router
from pydantic import BaseModel, Field

DEFAULT_INSTRUCTIONS = "Which option best matches the question?"

router = Router(preload=True)

app = FastAPI(title="Laya Crystalball")


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Input text to evaluate")
    options: List[str] = Field(..., min_length=2, description="Answer possibilities")
    instructions: Optional[str] = Field(None, description="Guidance for the decision")


class PredictResponse(BaseModel):
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    model: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    questions = {
        "answer": {
            "type": "choice",
            "instructions": req.instructions or DEFAULT_INSTRUCTIONS,
            "criteria": req.options,
        }
    }
    try:
        res = router.predict({"question": req.question}, questions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    ans = res["answers"]["answer"]
    return PredictResponse(
        choice=ans["choice"],
        confidence=ans["confidence"],
        probabilities=ans["probabilities"],
        model=res["routing"]["model"],
    )


@app.get("/health")
def health() -> Dict[str, object]:
    return {"status": "ok", "loaded": router.loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
