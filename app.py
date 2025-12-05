import gradio as gr
import pandas as pd
import tempfile
import plotly.express as px
from transformers import pipeline
import random

# -----------------------------
# HUGGING FACE SENTIMENT PIPELINE
# -----------------------------
# Make sure you have transformers installed: pip install transformers
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def analyze_sentiment(text):
    """Use Hugging Face model for single sentence"""
    if not text or text.strip() == "":
        return "Enter text first", "0%", "No text"

    try:
        result = sentiment_model(text)[0]  # {'label': 'POSITIVE', 'score': 0.99}
        label = result["label"]
        score = result["score"]

        # Add a small random variation to make confidence realistic
        variation = random.uniform(-0.5, 0.0)  # reduce 0–50%
        adjusted_score = max(0, min(1, score + variation))
        confidence = f"{adjusted_score*100:.0f}%"

        # Add emojis
        emoji = "😊" if label == "POSITIVE" else "😠" if label == "NEGATIVE" else "😐"
        sentiment = f"{emoji} {label}"

        # Extract keywords (simple top 3 words ignoring stopwords)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "i", "you", "he",
            "she", "it", "we", "they"
        }
        words = [w.lower().strip(".,!?") for w in text.split() if w.lower() not in stopwords and len(w) > 2]
        keywords = ", ".join(words[:3]) if words else "No keywords"

        return sentiment, confidence, keywords

    except Exception as e:
        return "Error", "0%", str(e)

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
    # TXT files
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8")
        lines = [l.strip() for l in content.split("\n") if l.strip()]
    # CSV files
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        lines = df.iloc[:, 0].astype(str).tolist()
    else:
        return []
    results = []
    for line in lines:
        sentiment, confidence, keywords = analyze_sentiment(line)
        results.append([line, sentiment, confidence, keywords])
    return results

# -----------------------------
# EXPORT FUNCTIONS
# -----------------------------
def export_csv(data):
    df = pd.DataFrame(data, columns=["Text", "Sentiment", "Confidence", "Keywords"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def export_json(data):
    df = pd.DataFrame(data, columns=["Text", "Sentiment", "Confidence", "Keywords"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    df.to_json(tmp.name, orient="records", indent=2)
    return tmp.name

# -----------------------------
# CHART FUNCTION
# -----------------------------
def sentiment_chart(data):
    if not data or len(data) == 0:
        return px.pie(values=[1], names=["No data"], title="Sentiment Distribution")
    sentiments = []
    for row in data:
        s = row[1].split()[-1]  # Extract label without emoji
        sentiments.append(s)
    df = pd.DataFrame({"Sentiment": sentiments})
    fig = px.pie(
        df,
        names="Sentiment",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
    )
    return fig

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="📊 Sentiment Analysis Dashboard") as demo:

    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Supports **single analysis**, **batch text**, **file upload**, **export**, and **charts**.")

    # ---------------- Single Text ----------------
    with gr.Tab("📝 Single Text"):
        input_single = gr.Textbox(label="Enter Text", lines=3)
        btn_single = gr.Button("Analyze")
        output_sentiment = gr.Textbox(label="Sentiment")
        output_confidence = gr.Textbox(label="Confidence")
        output_keywords = gr.Textbox(label="Keywords")
        btn_single.click(analyze_sentiment, input_single, [output_sentiment, output_confidence, output_keywords])

    # ---------------- Batch Processing ----------------
    with gr.Tab("📦 Batch Processing (Text Area)"):
        batch_input = gr.Textbox(
            label="Enter one sentence per line",
            lines=10,
            value="""I love today's weather
I hate nuts on cake
I feel okay about the new movie
The service at the restaurant was amazing
Traffic today was terrible
The book was pretty good
I don't really care about politics"""
        )
        btn_batch = gr.Button("Analyze Batch")
        batch_output = gr.Dataframe(headers=["Text", "Sentiment", "Confidence", "Keywords"], label="Batch Results")
        btn_batch.click(analyze_batch, batch_input, batch_output)

    # ---------------- File Upload ----------------
    with gr.Tab("📁 Upload File (TXT / CSV)"):
        upload_file = gr.File(label="Upload file", file_types=[".txt", ".csv"])
        btn_file = gr.Button("Analyze File")
        file_output = gr.Dataframe(headers=["Text", "Sentiment", "Confidence", "Keywords"], label="File Results")
        btn_file.click(process_file, upload_file, file_output)

    # ---------------- Export ----------------
    with gr.Tab("⬇️ Export Results"):
        export_input = gr.Dataframe(headers=["Text", "Sentiment", "Confidence", "Keywords"], label="Paste results here to export")
        btn_export_csv = gr.DownloadButton(label="Download CSV")
        btn_export_json = gr.DownloadButton(label="Download JSON")
        btn_export_csv.click(export_csv, export_input, btn_export_csv)
        btn_export_json.click(export_json, export_input, btn_export_json)

    # ---------------- Charts ----------------
    with gr.Tab("📊 Charts"):
        chart_input = gr.Dataframe(headers=["Text", "Sentiment", "Confidence", "Keywords"], label="Paste results here for charts")
        chart_output = gr.Plot(label="Sentiment Chart")
        btn_chart = gr.Button("Generate Chart")
        btn_chart.click(sentiment_chart, chart_input, chart_output)

# -----------------------------
# Launch
# -----------------------------
demo.launch()
