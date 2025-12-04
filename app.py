import gradio as gr
import requests
import os

# Get Hugging Face token from environment variable or use empty for public access
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# NEW: Updated API URL format
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

# Set headers - use token if available, otherwise public access
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face API"""
    if not text or not text.strip():
        return "Please enter text", "0%", "No input"
    
    try:
        # Make API request - NEW format
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": text,
                "parameters": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Print raw response
            print(f"API Response: {data}")
            
            # Process the response
            if isinstance(data, list) and len(data) > 0:
                predictions = data[0]
                
                # Find prediction with highest confidence
                best_prediction = max(predictions, key=lambda x: x['score'])
                
                # Map labels to readable format
                label_mapping = {
                    "LABEL_0": "NEGATIVE",
                    "LABEL_1": "POSITIVE",
                    "LABEL_2": "NEUTRAL",
                    "NEGATIVE": "NEGATIVE",
                    "POSITIVE": "POSITIVE",
                    "NEUTRAL": "NEUTRAL"
                }
                
                sentiment = label_mapping.get(best_prediction['label'], best_prediction['label'])
                confidence_score = best_prediction['score']
                confidence = f"{confidence_score:.2%}"
                
                # Simple keyword extraction
                words = text.lower().split()
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
                important_words = [word for word in words if word not in common_words and len(word) > 2]
                keywords = ", ".join(important_words[:5]) if important_words else "No significant keywords"
                
                return sentiment, confidence, keywords
            
            elif isinstance(data, dict) and 'error' in data:
                # Model is loading
                if 'loading' in data['error'].lower():
                    return "Model is loading...", "Please wait 20-30 seconds", "Refresh and try again"
                else:
                    return f"API Error", "0%", data['error'][:100]
            else:
                return "Unexpected response", "0%", str(data)[:100]
        
        elif response.status_code == 503:
            # Model is loading
            return "Model is loading...", "Please wait 20-30 seconds", "Refresh the page"
        
        elif response.status_code == 410:
            # Use alternative model
            return "Trying alternative model...", "0%", "Please wait"
        
        else:
            # Try with ALTERNATIVE MODEL
            alt_response = requests.post(
                "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
                headers=headers,
                json={"inputs": text},
                timeout=30
            )
            
            if alt_response.status_code == 200:
                alt_data = alt_response.json()
                if isinstance(alt_data, list) and len(alt_data) > 0:
                    predictions = alt_data[0]
                    best = max(predictions, key=lambda x: x['score'])
                    
                    # Map star ratings to sentiment
                    label = best['label']
                    if '1 star' in label or '2 stars' in label:
                        sentiment = "NEGATIVE"
                    elif '4 stars' in label or '5 stars' in label:
                        sentiment = "POSITIVE"
                    else:
                        sentiment = "NEUTRAL"
                    
                    confidence = f"{best['score']:.2%}"
                    
                    # Keywords
                    words = text.lower().split()
                    common = {'the', 'a', 'an', 'and', 'or', 'but', 'in'}
                    keywords = [w for w in words if w not in common and len(w) > 2][:3]
                    
                    return f"{sentiment} (Alt model)", confidence, ", ".join(keywords) if keywords else "None"
            
            return f"Error {response.status_code}", "0%", response.text[:100]
            
    except requests.exceptions.Timeout:
        return "Request timeout", "0%", "API took too long to respond"
    except requests.exceptions.RequestException as e:
        return "Connection Error", "0%", f"Check connection: {str(e)[:50]}"
    except Exception as e:
        return "Processing Error", "0%", f"Error: {str(e)[:50]}"

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
                placeholder="Type or paste your text here...\nExample: 'I love donuts! They make my mornings better.'",
                lines=3
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
        sentiment_output = gr.Textbox(label="Sentiment")
        confidence_output = gr.Textbox(label="Confidence")
        keywords_output = gr.Textbox(label="Keywords")
    
    # Examples section
    gr.Markdown("## 🎯 Examples")
    examples = gr.Examples(
        examples=[
            ["I love this! Fantastic experience."],
            ["Terrible product, I regret buying it."],
            ["It was okay, nothing special."]
        ],
        inputs=text_input,
        label="Click to test:"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center'><small>🚀 AI in Action Project | Built with Gradio & Hugging Face</small></div>")
    
    # Event handlers
    submit_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, keywords_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[text_input, sentiment_output, confidence_output, keywords_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, share=False)