# Sentiment Analysis API using FastAPI and Hugging Face Transformers

This repository contains a simple sentiment analysis API built with FastAPI and Hugging Face Transformers. The API uses a pretrained model to analyze the sentiment of given text input and returns whether the sentiment is positive, negative, or neutral along with the confidence score.

## Requirements

- Python 3.6+
- FastAPI
- Uvicorn
- Transformers (Hugging Face)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/NCINGA/idealize-24.git
   cd examples/sentiment_analysis-fastapi
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries:**

   ```bash
   pip install fastapi uvicorn transformers torch
   ```

## Running the API

1. **Start the FastAPI application using Uvicorn:**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.

## API Usage

### Endpoint

- **POST /analyze_sentiment**

  This endpoint analyzes the sentiment of the provided text.

  **Request Body:**

  ```json
  {
    "text": "I love using pretrained models for NLP!"
  }
  ```

  **Response:**

  ```json
  {
    "label": "POSITIVE",
    "score": 0.9998
  }
  ```

### Example Requests

#### Using curl

```bash
curl -X POST "http://127.0.0.1:8000/analyze_sentiment" -H "Content-Type: application/json" -d '{"text": "I love using pretrained models for NLP!"}'
```

#### Using Postman

1. Open Postman.
2. Create a new POST request.
3. Set the URL to `http://127.0.0.1:8000/analyze_sentiment`.
4. Set the request body to JSON and include a `text` field:
   ```json
   {
     "text": "I love using pretrained models for NLP!"
   }
   ```
5. Send the request and observe the response.


### Code Breakdown and Explanation

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
```
- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **BaseModel**: A class from `pydantic` library which is used for data validation and settings management using Python type annotations.
- **pipeline**: Function from the `transformers` library, used to create a pipeline for various tasks including sentiment analysis.
- **uvicorn**: An ASGI web server implementation for Python, used to run FastAPI applications.

```python
# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
```
- This line initializes a sentiment analysis pipeline. This uses models from the `transformers` library capable of analyzing the sentiment of text and classifying it as positive, negative, or neutral.

```python
# Define the request body structure
class TextRequest(BaseModel):
    text: str
```
- `TextRequest`: A Pydantic model that defines the expected structure of the request body. In this case, it expects a single field `text` which should be of type string.

```python
# Create the FastAPI app
app = FastAPI()
```
- Creates an instance of `FastAPI`. This object provides all the functionalities of FastAPI like creating routes, handling requests etc.

```python
@app.post("/analyze_sentiment")
def analyze_sentiment(request: TextRequest):
    # Get the text from the request
    text = request.text

    # Perform sentiment analysis
    result = sentiment_analyzer(text)[0]

    # Return the result
    return {"label": result["label"], "score": result["score"]}
```
- `@app.post("/analyze_sentiment")`: A route decorator that tells FastAPI that this function is responsible for handling POST requests to the `/analyze_sentiment` URL path.
- `analyze_sentiment(request: TextRequest)`: The function that gets called when there is a POST request to `/analyze_sentiment`. It expects a request body matching the `TextRequest` model.
- Inside the function, `text = request.text` extracts the text from the request.
- `sentiment_analyzer(text)[0]` performs the sentiment analysis on the provided text. The `pipeline` returns a list of results, and `[0]` is used to get the first result.
- Finally, the function returns a JSON object with the sentiment label and its associated confidence score.

```python
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
```
- Runs the app on the local development server using `uvicorn` with the specified host, port, and log level.

### Overall Functionality
This script sets up a basic API server that can receive text over HTTP and respond with sentiment analysis results. This is useful for applications needing to determine the sentiment of user-submitted text dynamically.
