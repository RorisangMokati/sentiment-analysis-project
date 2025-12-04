import gradio as gr
import requests
import os
import time

# === CONFIG ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODELS = [
    {"name": "distilbert-sst2",
     "url": "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
     "type": "sentiment"},

    {"name": "bert-stars",
     "url": "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
     "type": "stars"},

    {"name": "twitter-roberta",
     "url": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
     "type": "sentiment"},
]

# ======================================================================

def call_model(text, model):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        r = requests.post(model["url"], headers=headers, json={"inputs": text}, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"ERR {str(e)}"

def analyze_sentiment(text):
    if not text.strip():
        return "Enter text first", "0%", "No input"

    # === Try each model until one works ================================
    for i, model in enumerate(MODELS):
        if i > 0: time.sleep(0.4)
        ok, result = call_model(text, model)
        if ok:
            try:
                data = result[0]
                best = max(data, key=lambda x: x["score"])
                label = best["label"].upper()

                # SENTIMENT MAPPING
                if model["type"] == "sentiment":
                    sentiment = "POSITIVE" if "POS" in label or "LABEL_1" in label else \
                                "NEGATIVE" if "NEG" in label or "LABEL_0" in label else "NEUTRAL"
                else:  # star rating model
                    sentiment = "POSITIVE" if "4" in label or "5" in label else \
                                "NEGATIVE" if "1" in label or "2" in label else "NEUTRAL"

                confidence = f"{best['score']:.2%}"

                # keyword extraction
                words = [w for w in text.lower().split() if len(w) > 2][:5]
                keywords = ", ".join(words) if words else "None"

                return sentiment, confidence, keywords

            except:
                continue

    return rule_based(text)


# === RULE FALLBACK if HF down =========================================
def rule_based(text):
    text = text.lower()
    pos_words = ['love','good','great','awesome','excellent','best','happy']
    neg_words = ['bad','hate','terrible','worst','awful','sad','angry']

    pos = sum(w in text for w in pos_words)
    neg = sum(w in text for w in neg_words)

    if pos > neg: return "POSITIVE (Rule)", "70%", ", ".join(pos_words[:3])
    if neg > pos: return "NEGATIVE (Rule)", "70%", ", ".join(neg_words[:3])
    return "NEUTRAL (Rule)", "60%", "No strong keywords"


# ======================================================================
# ============================ UI ======================================
# ======================================================================

with gr.Blocks(title="Sentiment Analysis Dashboard") as demo:

    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Analyze emotional tone using AI with **multi-model fallback & offline rules**.")

    with gr.Row():
        with gr.Column(scale=2):

            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Type something like: I love donuts 😋",
                lines=4
            )

            with gr.Row():
                clear = gr.Button("🧹 Clear", variant="secondary")
                analyze = gr.Button("🚀 Analyze Sentiment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### ℹ How it Works")
            gr.Markdown("""
            • AI sentiment classification  
            • Multi-model fallback for reliability  
            • If API fails → rule-based sentiment kicks in  
            """)

    gr.Markdown("## 📈 Analysis Results")

    with gr.Row():
        sentiment = gr.Textbox(label="Sentiment", interactive=False)
        confidence = gr.Textbox(label="Confidence", interactive=False)
        keywords = gr.Textbox(label="Keywords", interactive=False)

    gr.Markdown("## 🎯 Try Examples")
    gr.Examples(
        [["I love this app, fantastic work!"]],
        [["Feeling sleepy 😴"]],
        [["Terrible product, waste of money."]],
        [["It was okay, nothing special really."]],
        [["Amazing service and support!"]],
        inputs=text_input
    )

    # Button events
    analyze.click(analyze_sentiment, text_input, [sentiment, confidence, keywords])
    clear.click(lambda:"", None, text_input)


# RUN APP
if __name__ == "__main__":
    demo.launch(debug=True, share=False)
