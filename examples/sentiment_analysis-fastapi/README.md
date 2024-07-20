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
   git clone https://github.com/your-username/sentiment-analysis-api.git
   cd sentiment-analysis-api
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries:**

   ```bash
   pip install fastapi uvicorn transformers
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

## Code Explanation

The `main.py` file contains the FastAPI application and the endpoint for sentiment analysis.

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

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
    return {
        "label": result['label'],
        "score": result['score']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
```

- **Initialization**: The sentiment analysis pipeline from Hugging Face Transformers is initialized.
- **TextRequest Model**: Defines the structure of the incoming request using Pydantic.
- **FastAPI App**: The FastAPI app is created, and an endpoint `/analyze_sentiment` is defined.
- **Endpoint Function**: This function takes a text input, performs sentiment analysis using the pipeline, and returns the result.
