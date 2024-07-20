from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


# Define request model
class Review(BaseModel):
    text: str


class ReviewsRequest(BaseModel):
    reviews: List[Review]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}


# Sentiment Analysis API endpoint
@app.post("/sentiment")
def analyze_sentiment(request: ReviewsRequest):
    results = [sentiment_analyzer(review.text)[0] for review in request.reviews]
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
