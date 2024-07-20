from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


class NLPModel:
    def __init__(self):
        # Sentiment analysis
        self.sa_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sa_tokenizer = AutoTokenizer.from_pretrained(self.sa_model_name)
        self.sa_model = AutoModelForSequenceClassification.from_pretrained(
            self.sa_model_name
        )

        # Named Entity Recognition
        self.ner_pipeline = pipeline(
            "ner", model="dbmdz/bert-large-cased-finetuned-conll03-english"
        )

        # Text Summarization
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def predict_sentiment(self, text):
        inputs = self.sa_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.sa_model(**inputs)

        predicted_class = torch.argmax(outputs.logits).item()
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][
            predicted_class
        ].item()

        return {
            "sentiment": "positive" if predicted_class == 1 else "negative",
            "confidence": confidence,
        }

    def named_entity_recognition(self, text):
        results = self.ner_pipeline(text)
        entities = {}
        for result in results:
            entity_type = result["entity"]
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(result["word"])
        return entities

    def summarize_text(self, text, max_length=150, min_length=50):
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return summary[0]["summary_text"]
