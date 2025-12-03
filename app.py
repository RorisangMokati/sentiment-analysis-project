import gradio as gr
import requests
import os

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
KEYWORD_MODEL = "ml6team/keyphrase-extraction-kbir"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def single_text_analysis(text):
    if not text.strip():
        return "No text provided", "", ""

    try:
        # Sentiment
        sentiment_resp = requests.post(
            f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}",
            headers=headers,
            json={"inputs": text}
        )
        sentiment_resp.raise_for_status()  # raise error if request fails
        sentiment_data = sentiment_resp.json()
        sentiment = sentiment_data[0][0]['label']
        score = sentiment_data[0][0]['score']

        # Keywords
        keyword_resp = requests.post(
            f"https://api-inference.huggingface.co/models/{KEYWORD_MODEL}",
            headers=headers,
            json={"inputs": text}
        )
        keyword_resp.raise_for_status()
        keywords_data = keyword_resp.json()
        keywords = [k['text'] for k in keywords_data[0]]

        return sentiment, f"{score:.2f}", ", ".join(keywords)

    except Exception as e:
        return "Error", "Error", str(e)


demo = gr.Interface(
    fn=single_text_analysis,
    inputs=gr.Textbox(lines=4, label="Enter text"),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Keywords")
    ]
)
demo.launch()
