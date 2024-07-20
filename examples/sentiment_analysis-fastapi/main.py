from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


# Define the request body structure
class TextRequest(BaseModel):
    text: str


# Create the FastAPI app
app = FastAPI()


@app.post("/analyze_sentiment")
def analyze_sentiment(request: TextRequest):
    # Get the text from the request
    text = request.text

    # Perform sentiment analysis
    result = sentiment_analyzer(text)[0]

    # Return the result
    return {"label": result["label"], "score": result["score"]}


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
