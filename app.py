import gradio as gr
import requests

HF_API_TOKEN = "YOUR_HF_TOKEN"  # <-- put your Hugging Face API token here
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
KEYWORD_MODEL = "ml6team/keyphrase-extraction-kbir"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def single_text_analysis(text):
    if not text.strip():
        return "No text provided", "", ""

    # Sentiment analysis
    sentiment_resp = requests.post(
        f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}",
        headers=headers,
        json={"inputs": text}
    )
    sentiment = sentiment_resp.json()[0][0]['label']
    score = sentiment_resp.json()[0][0]['score']

    # Keyword extraction
    keyword_resp = requests.post(
        f"https://api-inference.huggingface.co/models/{KEYWORD_MODEL}",
        headers=headers,
        json={"inputs": text}
    )
    keywords = [k['text'] for k in keyword_resp.json()[0]]

    return sentiment, f"{score:.2f}", ", ".join(keywords)

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
