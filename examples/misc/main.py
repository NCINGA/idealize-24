from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import NLPModel

app = FastAPI(
    title="NLP Tasks API",
    description="API for various NLP tasks including sentiment analysis, named entity recognition, and text summarization.",
    version="1.0.0",
)

nlp_model = NLPModel()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The input text for NLP analysis")


class SummarizationRequest(TextRequest):
    max_length: int = Field(
        150, ge=50, le=500, description="Maximum length of the summary"
    )
    min_length: int = Field(
        50, ge=10, le=100, description="Minimum length of the summary"
    )


@app.post("/predict_sentiment", tags=["Sentiment Analysis"])
async def sentiment_analysis(request: TextRequest):
    """
    Predict the sentiment of the input text.
    Returns sentiment (positive/negative) and confidence score.
    """
    try:
        result = nlp_model.predict_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/named_entity_recognition", tags=["Named Entity Recognition"])
async def ner(request: TextRequest):
    """
    Perform Named Entity Recognition on the input text.
    Returns a dictionary of entity types and corresponding entities.
    """
    try:
        result = nlp_model.named_entity_recognition(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", tags=["Text Summarization"])
async def summarize(request: SummarizationRequest):
    """
    Generate a summary of the input text.
    Returns the summarized text.
    """
    try:
        summary = nlp_model.summarize_text(
            request.text, request.max_length, request.min_length
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
