# sentiment_analyzer.py
from transformers import pipeline

def analyze_sentiment(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    return result[0]['label'], result[0]['score']

if __name__ == "__main__":
    sample = "I love how intuitive AI tools are becoming!"
    label, score = analyze_sentiment(sample)
    print(f"Sentiment: {label} ({score:.2f})")
