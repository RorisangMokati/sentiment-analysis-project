import gradio as gr
import requests
import os
import time

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# LIST OF WORKING MODELS (try in order)
MODELS = [
    {
        "name": "distilbert-sst2",
        "url": "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
        "type": "sentiment"
    },
    {
        "name": "bert-sentiment",
        "url": "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
        "type": "stars"
    },
    {
        "name": "twitter-roberta",
        "url": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
        "type": "sentiment"
    }
]

def analyze_with_model(text, model_config):
    """Try to analyze with a specific model"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    try:
        response = requests.post(
            model_config["url"],
            headers=headers,
            json={"inputs": text},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, f"Status {response.status_code}"
    except Exception as e:
        return False, str(e)

def analyze_sentiment(text):
    """Main sentiment analysis function with multiple fallbacks"""
    if not text or not text.strip():
        return "Please enter text", "0%", "No input"
    
    # Try each model in order
    for i, model in enumerate(MODELS):
        if i > 0:
            time.sleep(0.5)  # Small delay between tries
        
        success, result = analyze_with_model(text, model)
        
        if success:
            # Process based on model type
            if model["type"] == "sentiment":
                if isinstance(result, list) and len(result) > 0:
                    predictions = result[0]
                    best = max(predictions, key=lambda x: x['score'])
                    
                    # Map labels
                    label = best['label'].upper()
                    if "POSITIVE" in label or "LABEL_1" in label:
                        sentiment = "POSITIVE"
                    elif "NEGATIVE" in label or "LABEL_0" in label:
                        sentiment = "NEGATIVE"
                    else:
                        sentiment = "NEUTRAL"
                    
                    confidence = f"{best['score']:.2%}"
                    
            elif model["type"] == "stars":
                if isinstance(result, list) and len(result) > 0:
                    predictions = result[0]
                    best = max(predictions, key=lambda x: x['score'])
                    
                    # Map star ratings
                    label = best['label']
                    if '1 star' in label or '2 stars' in label:
                        sentiment = "NEGATIVE"
                    elif '4 stars' in label or '5 stars' in label:
                        sentiment = "POSITIVE"
                    else:
                        sentiment = "NEUTRAL"
                    
                    confidence = f"{best['score']:.2%}"
            
            # Extract keywords
            words = text.lower().split()
            common = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
            keywords = [w for w in words if w not in common and len(w) > 2][:3]
            
            return sentiment, confidence, ", ".join(keywords) if keywords else "None"
    
    # If all models fail, use RULE-BASED fallback
    return analyze_with_rules(text)

def analyze_with_rules(text):
    """Rule-based sentiment analysis as final fallback"""
    text_lower = text.lower()
    
    # Positive words
    positive_words = ['love', 'like', 'good', 'great', 'excellent', 'fantastic', 'awesome', 'best', 'happy', 'perfect', 'wonderful']
    # Negative words
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'poor', 'disappointed', 'regret', 'sorry', 'unhappy']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = f"{(70 + min(pos_count * 10, 25)):.0f}%"
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = f"{(70 + min(neg_count * 10, 25)):.0f}%"
    else:
        sentiment = "NEUTRAL"
        confidence = "65%"
    
    # Extract actual words found
    found_words = []
    all_keywords = positive_words + negative_words
    for word in all_keywords:
        if word in text_lower:
            found_words.append(word)
    
    keywords = ", ".join(found_words[:3]) if found_words else "No keywords detected"
    
    return f"{sentiment} (Rule-based)", confidence, keywords

# Create the Gradio interface
with gr.Blocks(title="Sentiment Analysis Dashboard") as demo:
    
    # Header
    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Analyze emotional tone using AI sentiment classification.")
    
    # Main content area
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Type or paste your text here...",
                lines=3,
                value="I love this! Fantastic experience."
            )
            
            # Buttons
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Analyze Sentiment", variant="primary")
        
        with gr.Column(scale=1):
            # Information panel
            gr.Markdown("### ℹ️ How it works")
            gr.Markdown("""
            1. **Enter any sentence**
            2. **Click Analyze Sentiment**
            3. **View Sentiment, Confidence & Keywords**
            """)
    
    # Results section
    gr.Markdown("## 📈 Results")
    
    with gr.Row():
        sentiment_output = gr.Textbox(
            label="Sentiment",
            value="Click 'Analyze' to see results"
        )
    
    with gr.Row():
        confidence_output = gr.Textbox(
            label="Confidence",
            value=""
        )
        keywords_output = gr.Textbox(
            label="Keywords", 
            value=""
        )
    
    # Status indicator
    status = gr.Textbox(
        label="Status",
        value="Ready",
        interactive=False
    )
    
    # Examples section
    gr.Markdown("## 🎯 Examples")
    examples = gr.Examples(
        examples=[
            ["I love this! Fantastic experience."],
            ["Terrible product, I regret buying it."],
            ["It was okay, nothing special."],
            ["The service was excellent and fast!"],
            ["Worst customer experience ever."]
        ],
        inputs=text_input,
        label="Click to test:"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center'><small>🚀 AI in Action Project | Built with Gradio & Hugging Face</small></div>")
    
    # Event handlers
    def process_with_status(text):
        """Process with status updates"""
        status.update(value="🔄 Analyzing...")
        sentiment, confidence, keywords = analyze_sentiment(text)
        status.update(value="✅ Analysis complete!")
        return sentiment, confidence, keywords
    
    submit_btn.click(
        fn=process_with_status,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, keywords_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "Click 'Analyze' to see results", "", "", "Ready"),
        outputs=[text_input, sentiment_output, confidence_output, keywords_output, status]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, share=False)