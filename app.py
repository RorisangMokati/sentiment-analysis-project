import gradio as gr
import csv
import json
import pandas as pd
from io import StringIO
 
# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def analyze_sentiment(text):
    """Simple sentiment logic."""
    if not text or text.strip() == "":
        return "Enter text first", "0%", "No text"
 
    text_lower = text.lower()
 
    positive_words = ["love", "fantastic", "excellent"]
    negative_words = ["hate", "terrible", "worst"]
    neutral_words  = ["okay", "fine", "average"]
 
    matched = [w for w in positive_words + negative_words + neutral_words if w in text_lower]
 
    if any(w in text_lower for w in positive_words):
        return "😊 POSITIVE", "94%", ", ".join(matched)
    elif any(w in text_lower for w in negative_words):
        return "😠 NEGATIVE", "89%", ", ".join(matched)
    elif any(w in text_lower for w in neutral_words):
        return "😐 NEUTRAL", "76%", ", ".join(matched)
 
    if len(text.split()) > 5:
        return "🤔 NEUTRAL", "68%", "multiple words detected"
    else:
        return "😐 NEUTRAL", "72%", "short text"
 
 
# -----------------------------
# BATCH PROCESSING
# -----------------------------
def analyze_batch(input_text):
    if not input_text or input_text.strip() == "":
        return []
 
    lines = [l.strip() for l in input_text.split("\n") if l.strip()]
    results = []
 
    for line in lines:
        sentiment, confidence, keywords = analyze_sentiment(line)
        results.append([line, sentiment, confidence, keywords])
 
    return results
 
 
# -----------------------------
# FILE UPLOAD PROCESSING
# -----------------------------
def process_file(file):
    if file is None:
        return []
 
    # Detect file type
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8")
        lines = content.split("\n")
 
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file.name)
        if df.shape[1] == 0:
            return []
        lines = df.iloc[:, 0].astype(str).tolist()
 
    else:
        return []
 
    results = []
    for line in lines:
        line = line.strip()
        if line:
            sentiment, confidence, keywords = analyze_sentiment(line)
            results.append([line, sentiment, confidence, keywords])
 
    return resul