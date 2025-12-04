import gradio as gr
import requests
import os

# Get Hugging Face token from environment variable or use empty for public access
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face API"""
    if not text or not text.strip():
        return "Please enter text", "0%", "No input"
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                predictions = data[0]
                best = max(predictions, key=lambda x: x['score'])
                
                label_map = {"LABEL_0": "NEGATIVE","LABEL_1": "POSITIVE"}
                sentiment = label_map.get(best['label'], best['label'])
                confidence = f"{best['score']:.2%}"
                
                words = text.lower().split()
                common = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were',
                          'i','you','he','she','it','we','they'}
                keywords = ", ".join([w for w in words if w not in common and len(w)>2][:5]) or "No significant keywords"
                
                return sentiment, confidence, keywords

        elif response.status_code == 503:
            return "Model loading...", "wait 10-20s", "Try again soon"

        return f"API Error {response.status_code}", "0%", response.text[:80]

    except Exception as e:
        return "Error", "0%", str(e)[:80]


def process_text(text):
    return analyze_sentiment(text)


# Custom CSS moved into <style> (works in newer Gradio)
custom_css = """
.gr-button-primary {
    background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
    border: none !important;
}
.gr-box { border-radius: 10px !important; }
"""

with gr.Blocks(title="Sentiment Analysis Dashboard") as demo:
    
    # Inject CSS here
    gr.HTML(f"<style>{custom_css}</style>")

    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Analyze emotional tone using AI sentiment classification.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Type text here...",
                lines=4
            )

            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Analyze Sentiment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### ℹ️ How it works")
            gr.Markdown("""
1. Enter any sentence  
2. Click **Analyze Sentiment**  
3. View **Sentiment**, **Confidence** & **Keywords**
""")

    gr.Markdown("## 📈 Results")

    with gr.Row():
        sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
        confidence_output = gr.Textbox(label="Confidence", interactive=False)
        keywords_output = gr.Textbox(label="Keywords", interactive=False)

    gr.Examples(
        examples=[
            ["I love this! Fantastic experience."],
            ["Terrible product, I regret buying it."],
            ["It was okay, nothing special."],
        ],
        inputs=text_input,
    )

    submit_btn.click(fn=process_text, inputs=text_input,
                     outputs=[sentiment_output, confidence_output, keywords_output])

    clear_btn.click(fn=lambda: ("", "", ""), 
                    outputs=[sentiment_output, confidence_output, keywords_output,])


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
