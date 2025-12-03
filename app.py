import gradio as gr
import requests
import os
from huggingface_hub import InferenceClient


HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
KEYWORD_MODEL = "ml6team/keyphrase-extraction-kbir"

INFERENCE_API_URL = "https://router.huggingface.co/hf-inference"


def single_text_analysis(text):
    if not text.strip():
        return "No text provided", "", ""

    try:
        # Sentiment analysis
        sentiment_resp = requests.post(
            f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}",
            headers=headers,
            json={"inputs": text}
        )
        sentiment_resp.raise_for_status()
        sentiment_data = sentiment_resp.json()
        # Check if the API returned an error message instead of list
        if isinstance(sentiment_data, dict) and sentiment_data.get("error"):
            return "Model loading, try again...", "", ""
        sentiment = sentiment_data[0][0]['label']
        score = sentiment_data[0][0]['score']

        # Keyword extraction
        keyword_resp = requests.post(
            f"https://api-inference.huggingface.co/models/{KEYWORD_MODEL}",
            headers=headers,
            json={"inputs": text}
        )
        keyword_resp.raise_for_status()
        keywords_data = keyword_resp.json()
        if isinstance(keywords_data, dict) and keywords_data.get("error"):
            keywords = []
        else:
            keywords = [k['text'] for k in keywords_data[0]]

        return sentiment, f"{score:.2f}", ", ".join(keywords)

    except Exception as e:
        return "Error", "Error", str(e)

# Gradio interface
demo = gr.Interface(
    fn=single_text_analysis,
    inputs=gr.Textbox(lines=4, label="Enter text"),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Keywords")
    ],
    title="Sentiment Analysis Dashboard",
    description="Analyze the sentiment of your text and extract keywords using Hugging Face models."
)

demo.launch()
