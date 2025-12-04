import gradio as gr
import json
import pandas as pd
import tempfile

# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def analyze_sentiment(text):
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

    # TXT files
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8")
        lines = [l.strip() for l in content.split("\n") if l.strip()]

    # CSV files
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)   # use uploaded file
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
    with open(tmp.name, "w") as f:
        f.write(df.to_json(orient="records", indent=2))
    return tmp.name

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="📊 Sentiment Analysis Dashboard") as demo:

    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Supports **single analysis**, **batch text**, **file upload**, and **multiple export options**.")

    # ---------------- Single Text ----------------
    with gr.Tab("📝 Single Text"):
        input_single = gr.Textbox(label="Enter Text", lines=3)
        btn_single = gr.Button("Analyze")

        output_sentiment = gr.Textbox(label="Sentiment")
        output_confidence = gr.Textbox(label="Confidence")
        output_keywords = gr.Textbox(label="Keywords")

        btn_single.click(analyze_sentiment, input_single,
                         [output_sentiment, output_confidence, output_keywords])

    # ---------------- Batch Processing ----------------
    with gr.Tab("📦 Batch Processing (Text Area)"):
        batch_input = gr.Textbox(
            label="Enter one sentence per line",
            lines=6,
            placeholder="I love this!\nTerrible product.\nIt was fine."
        )
        btn_batch = gr.Button("Analyze Batch")

        batch_output = gr.Dataframe(
            headers=["Text", "Sentiment", "Confidence", "Keywords"],
            label="Batch Results"
        )

        btn_batch.click(analyze_batch, batch_input, batch_output)

    # ---------------- File Upload ----------------
    with gr.Tab("📁 Upload File (TXT / CSV)"):
        upload_file = gr.File(label="Upload file", file_types=[".txt", ".csv"])
        btn_file = gr.Button("Analyze File")

        file_output = gr.Dataframe(
            headers=["Text", "Sentiment", "Confidence", "Keywords"],
            label="File Results"
        )

        btn_file.click(process_file, upload_file, file_output)

    # ---------------- Export ----------------
    with gr.Tab("⬇️ Export Results"):
        export_input = gr.Dataframe(
            headers=["Text", "Sentiment", "Confidence", "Keywords"],
            label="Paste results here to export"
        )

        # FIXED: removed file_name argument
        btn_export_csv = gr.DownloadButton(label="Download CSV")
        btn_export_json = gr.DownloadButton(label="Download JSON")

        btn_export_csv.click(export_csv, export_input, btn_export_csv)
        btn_export_json.click(export_json, export_input, btn_export_json)

# -----------------------------
# Launch
# -----------------------------
demo.launch()
