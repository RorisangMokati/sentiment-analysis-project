import gradio as gr
import requests
import os

# Get Hugging Face token from environment variable or use empty for public access
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

# Set headers - use token if available, otherwise public access
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face API"""
    if not text or not text.strip():
        return {
            "sentiment": "Please enter text",
            "confidence": "0%",
            "keywords": "No input"
        }
    
    try:
        # Make API request
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the response
            if isinstance(data, list) and len(data) > 0:
                predictions = data[0]
                
                # Find prediction with highest confidence
                best_prediction = max(predictions, key=lambda x: x['score'])
                
                # Map labels to readable format
                label_mapping = {
                    "LABEL_0": "NEGATIVE",
                    "LABEL_1": "POSITIVE",
                    "NEGATIVE": "NEGATIVE",
                    "POSITIVE": "POSITIVE"
                }
                
                sentiment = label_mapping.get(best_prediction['label'], best_prediction['label'])
                confidence_score = best_prediction['score']
                confidence = f"{confidence_score:.2%}"
                
                # Simple keyword extraction
                words = text.lower().split()
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
                important_words = [word for word in words if word not in common_words and len(word) > 2]
                keywords = ", ".join(important_words[:5]) if important_words else "No significant keywords"
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "keywords": keywords
                }
            else:
                return {
                    "sentiment": "Invalid response",
                    "confidence": "0%",
                    "keywords": f"Response: {str(data)[:100]}"
                }
        
        elif response.status_code == 503:
            # Model is loading
            return {
                "sentiment": "Model is loading...",
                "confidence": "Please wait 10-20 seconds",
                "keywords": "Try again in a moment"
            }
        
        else:
            # Other API errors
            return {
                "sentiment": f"API Error {response.status_code}",
                "confidence": "0%",
                "keywords": f"Details: {response.text[:100]}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "sentiment": "Connection Error",
            "confidence": "0%",
            "keywords": f"Check internet: {str(e)[:50]}"
        }
    except Exception as e:
        return {
            "sentiment": "Processing Error",
            "confidence": "0%",
            "keywords": f"Error: {str(e)[:50]}"
        }

def process_text(text):
    """Process text and return formatted results"""
    result = analyze_sentiment(text)
    return result["sentiment"], result["confidence"], result["keywords"]

# Custom CSS for better appearance
custom_css = """
.gr-button-primary {
    background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
    border: none !important;
}
.gr-box {
    border-radius: 10px !important;
}
"""

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis Dashboard", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # Header
    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Analyze the emotional tone of text using AI. Enter text below to get sentiment, confidence score, and keywords.")
    
    # Main content area
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            text_input = gr.Textbox(
                label="Enter Text",
                placeholder="Type or paste your text here...\nExample: 'I love donuts! They make my mornings better.'",
                lines=4,
                elem_id="text_input"
            )
            
            # Buttons
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", variant="secondary", size="sm")
                submit_btn = gr.Button("🚀 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Information panel
            gr.Markdown("### ℹ️ How it works:")
            gr.Markdown("""
            1. **Enter text** in the box
            2. Click **Analyze Sentiment**
            3. View **sentiment** (Positive/Negative)
            4. Check **confidence** score
            5. See **keywords** that influenced the analysis
            """)
    
    # Results section
    gr.Markdown("## 📈 Analysis Results")
    with gr.Row():
        with gr.Column():
            sentiment_output = gr.Textbox(
                label="Sentiment",
                interactive=False,
                elem_id="sentiment_box"
            )
        with gr.Column():
            confidence_output = gr.Textbox(
                label="Confidence Score",
                interactive=False,
                elem_id="confidence_box"
            )
        with gr.Column():
            keywords_output = gr.Textbox(
                label="Keywords",
                interactive=False,
                elem_id="keywords_box"
            )
    
    # Examples section
    gr.Markdown("## 🎯 Try These Examples:")
    examples = gr.Examples(
        examples=[
            ["I absolutely love this product! It exceeded all my expectations and works perfectly."],
            ["Terrible experience. The service was slow and the staff was unhelpful."],
            ["The movie was okay, not great but not bad either. Decent entertainment."],
            ["This is the best day ever! Everything went perfectly from morning till night."],
            ["I'm very disappointed with the quality. Broke after just one week of use."]
        ],
        inputs=text_input,
        label="Click any example to test:"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center'>
        <p>🚀 <b>AI in Action Project</b> | Built with Gradio & Hugging Face</p>
        <p><small>Model: distilbert-base-uncased-finetuned-sst-2-english</small></p>
    </div>
    """)
    
    # Event handlers
    submit_btn.click(
        fn=process_text,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, keywords_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[text_input, sentiment_output, confidence_output, keywords_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=False
    )