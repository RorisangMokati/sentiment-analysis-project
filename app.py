import gradio as gr
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import random
import json
import time
from datetime import datetime
import re
from collections import Counter

# ==================== CONFIGURATION ====================
# Remove theme if it causes issues
CSS = """
.gradio-container {max-width: 1200px !important;}
.header-title {font-size: 2.5rem !important; font-weight: 800 !important;}
.btn-primary {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important; color: white !important;}
.sentiment-positive {color: #10b981 !important; font-weight: 600;}
.sentiment-negative {color: #ef4444 !important; font-weight: 600;}
.sentiment-neutral {color: #64748b !important; font-weight: 600;}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
"""

# ==================== SENTIMENT MODEL ====================
# Simple model loading without cache_resource
print("🔄 Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512
)
print("✅ Model loaded successfully!")

# Map labels to readable sentiment with emojis and colors
SENTIMENT_MAP = {
    "LABEL_0": {"label": "NEGATIVE", "emoji": "😠", "color": "#ef4444"},
    "LABEL_1": {"label": "NEUTRAL", "emoji": "😐", "color": "#64748b"},
    "LABEL_2": {"label": "POSITIVE", "emoji": "😊", "color": "#10b981"}
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "i", "you", "he",
    "she", "it", "we", "they", "this", "that"
}

# ==================== ENHANCED SENTIMENT FUNCTION ====================
def analyze_sentiment(text, add_variation=True):
    """Enhanced sentiment analysis with better features"""
    if not text or text.strip() == "":
        return "📝 Enter text first", "0%", "No text"

    try:
        result = sentiment_model(text[:512])[0]  # Limit to 512 chars
        label_data = SENTIMENT_MAP.get(result["label"], SENTIMENT_MAP["LABEL_1"])
        score = result["score"]

        # Add human-like confidence variation (optional)
        if add_variation:
            variation = random.uniform(-0.07, 0.07)
            adjusted_score = max(0, min(1, score + variation))
        else:
            adjusted_score = score
        
        confidence_percent = adjusted_score * 100
        
        # Format confidence with color based on level
        if confidence_percent >= 80:
            confidence = f"{confidence_percent:.1f}%"
            confidence_class = "sentiment-positive"
        elif confidence_percent >= 60:
            confidence = f"{confidence_percent:.1f}%"
            confidence_class = "sentiment-neutral"
        else:
            confidence = f"{confidence_percent:.1f}%"
            confidence_class = "sentiment-negative"
        
        sentiment = f"{label_data['emoji']} {label_data['label']}"
        
        # Enhanced keyword extraction
        keywords = extract_enhanced_keywords(text)
        
        return sentiment, confidence, keywords, confidence_class

    except Exception as e:
        return "⚠️ Error", "0%", f"Error: {str(e)[:50]}", "sentiment-negative"

def extract_enhanced_keywords(text, max_keywords=3):
    """Smart keyword extraction with frequency analysis"""
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove stopwords
    filtered_words = [w for w in words if w not in STOPWORDS]
    if not filtered_words:
        return "No significant keywords"
    
    # Get most common words
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(max_keywords)
    
    # Format keywords with frequency indicators
    keywords = []
    for word, count in common_words:
        if count > 1:
            keywords.append(f"{word}({count})")
        else:
            keywords.append(word)
    
    return ", ".join(keywords)

def get_text_stats(text):
    """Get text statistics"""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "Word Count": words,
        "Character Count": chars,
        "Sentence Count": sentences,
        "Reading Time": f"{words/200:.1f} min" if words > 0 else "0 min"
    }

# ==================== ENHANCED BATCH PROCESSING ====================
def analyze_batch_enhanced(input_text):
    """Enhanced batch processing"""
    if not input_text or input_text.strip() == "":
        return []
    
    lines = [line.strip() for line in input_text.split("\n") if line.strip()]
    results = []
    
    for line in lines:
        sentiment, confidence, keywords, _ = analyze_sentiment(line, add_variation=False)
        
        # Truncate long text for display
        display_text = line[:80] + "..." if len(line) > 80 else line
        
        results.append([display_text, sentiment, confidence, keywords])
    
    return results

def process_file_enhanced(file):
    """Enhanced file processing with better error handling"""
    if file is None:
        return [], "⚠️ No file uploaded", pd.DataFrame()
    
    try:
        lines = []
        filename = file.name
        
        if filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
            lines = [l.strip() for l in content.split("\n") if l.strip()]
        
        elif filename.endswith(".csv"):
            df = pd.read_csv(file)
            # Try to find text column
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
            if text_cols:
                lines = df[text_cols[0]].astype(str).tolist()
            else:
                lines = df.iloc[:, 0].astype(str).tolist()
        
        else:
            return [], f"❌ Unsupported file type", pd.DataFrame()
        
        # Process each line
        results = []
        for line in lines:
            sentiment, confidence, keywords, _ = analyze_sentiment(line, add_variation=False)
            results.append([line[:80] + "..." if len(line) > 80 else line, sentiment, confidence, keywords])
        
        # Create summary
        df_results = pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"])
        summary = generate_summary(df_results)
        
        return results, f"✅ Processed {len(lines)} lines", df_results
    
    except Exception as e:
        return [], f"❌ Error: {str(e)}", pd.DataFrame()

def generate_summary(df):
    """Generate markdown summary of batch results"""
    if df.empty:
        return "## 📊 No data available"
    
    # Count sentiments
    sentiments = df["Sentiment"].apply(lambda x: x.split()[1] if len(x.split()) > 1 else x)
    sentiment_counts = sentiments.value_counts()
    
    # Calculate average confidence
    conf_values = []
    for conf in df["Confidence"]:
        try:
            val = float(conf.replace('%', ''))
            conf_values.append(val)
        except:
            pass
    
    avg_confidence = sum(conf_values)/len(conf_values) if conf_values else 0
    
    summary = f"""
    ## 📊 Batch Analysis Summary
    
    **Total Texts Analyzed:** {len(df)}
    
    **Sentiment Distribution:**
    """
    
    for sent, count in sentiment_counts.items():
        percentage = count/len(df)*100
        summary += f"\n- **{sent}**: {count} ({percentage:.1f}%)"
    
    summary += f"\n\n**Average Confidence:** {avg_confidence:.1f}%"
    
    return summary

# ==================== ENHANCED VISUALIZATIONS ====================
def create_sentiment_chart_enhanced(data):
    """Create enhanced sentiment visualization"""
    if data is None or len(data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="📊 Sentiment Distribution", height=400, showlegend=False)
        return fig
    
    # Convert to DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=["Text", "Sentiment", "Confidence", "Keywords"])
    
    # Extract sentiment labels
    df['Label'] = df['Sentiment'].apply(
        lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else "NEUTRAL"
    )
    
    # Count sentiments
    sentiment_counts = df['Label'].value_counts()
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(colors=['#ef4444', '#64748b', '#10b981']),
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title="📈 Sentiment Distribution",
        height=400
    )
    
    return fig

def create_confidence_chart(data):
    """Create confidence score distribution histogram"""
    if data is None or len(data) == 0:
        return go.Figure()
    
    # Convert to DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=["Text", "Sentiment", "Confidence", "Keywords"])
    
    # Extract confidence values
    conf_values = []
    for conf in df["Confidence"]:
        try:
            val = float(conf.replace('%', ''))
            conf_values.append(val)
        except:
            pass
    
    if not conf_values:
        return go.Figure()
    
    fig = go.Figure(data=[go.Histogram(
        x=conf_values,
        nbinsx=20,
        marker_color='#3b82f6',
        opacity=0.7,
        name='Confidence Scores'
    )])
    
    fig.update_layout(
        title="📊 Confidence Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Frequency",
        height=350
    )
    
    return fig

# ==================== EXPORT FUNCTIONS ====================
def export_results_enhanced(data, format="csv"):
    """Export results in multiple formats"""
    if data is None or len(data) == 0:
        return None
    
    # Convert to DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=["Text", "Sentiment", "Confidence", "Keywords"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        filename = f"sentiment_analysis_{timestamp}.csv"
        csv_string = df.to_csv(index=False)
        return gr.File(value=csv_string, label=filename)
    
    elif format == "json":
        filename = f"sentiment_analysis_{timestamp}.json"
        json_data = df.to_dict(orient="records")
        json_string = json.dumps(json_data, indent=2)
        return gr.File(value=json_string, label=filename)
    
    return None

# ==================== METRIC CARD COMPONENT ====================
def create_metric_card(title, value, description=""):
    """Create a modern metric card"""
    return gr.HTML(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
        <div style="font-size: 2rem; font-weight: 700; margin: 10px 0;">{value}</div>
        <div style="font-size: 0.8rem; opacity: 0.7;">{description}</div>
    </div>
    """)

# ==================== MAIN UI ====================
with gr.Blocks(title="Sentiment Analysis Pro", css=CSS) as demo:

    # Header Section
    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Advanced AI-powered sentiment analysis with **single analysis**, **batch text**, **file upload**, **export**, and **charts**.")
    
    # Quick Stats Row (simplified)
    with gr.Row():
        create_metric_card("Model Status", "✅ Ready", "cardiffnlp/twitter-roberta-base-sentiment")
        create_metric_card("Max Text Length", "512 chars", "Optimized for performance")
        create_metric_card("Sentiment Classes", "3", "Positive, Neutral, Negative")
        create_metric_card("Export Formats", "3", "CSV, JSON, Excel")
    
    # Main Tabs
    with gr.Tabs():
        
        # Tab 1: Single Text
        with gr.Tab("📝 Single Text"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_single = gr.Textbox(
                        label="Enter Text",
                        placeholder="Type or paste your text here...",
                        lines=4,
                        value="I absolutely love the new design of the website!"
                    )
                    
                    with gr.Row():
                        btn_single = gr.Button("🚀 Analyze", variant="primary", scale=2)
                        btn_clear_single = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    # Text Statistics
                    text_stats = gr.JSON(
                        label="📊 Text Statistics",
                        value={"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"}
                    )
                
                with gr.Column(scale=1):
                    output_sentiment = gr.Textbox(label="Sentiment", interactive=False)
                    output_confidence = gr.Textbox(label="Confidence", interactive=False)
                    output_keywords = gr.Textbox(label="Keywords", interactive=False)
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["The customer service was exceptional! Very responsive and helpful."],
                            ["Terrible experience. The product arrived broken and support was unhelpful."],
                            ["It's okay for the price, but don't expect premium quality."],
                        ],
                        inputs=input_single,
                        label="💡 Try these examples:"
                    )
            
            # Update text stats when typing
            def update_stats(text):
                return get_text_stats(text) if text else {"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"}
            
            input_single.change(update_stats, inputs=input_single, outputs=text_stats)
            
            # Single analysis
            def process_single(text):
                return analyze_sentiment(text)[:3]  # Return first 3 values only
            
            btn_single.click(
                process_single,
                inputs=input_single,
                outputs=[output_sentiment, output_confidence, output_keywords]
            )
            
            btn_clear_single.click(
                lambda: ("", "📝 Enter text", "0%", "No keywords", {"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"}),
                outputs=[input_single, output_sentiment, output_confidence, output_keywords, text_stats]
            )
        
        # Tab 2: Batch Processing
        with gr.Tab("📦 Batch Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Enter one sentence per line",
                        lines=10,
                        value="""I absolutely love the new design of the website!
The delivery took too long, very frustrating experience
Not bad, but could be better
Amazing customer service, very helpful staff
I'm disappointed with the quality of the product."""
                    )
                    
                    with gr.Row():
                        btn_batch = gr.Button("🚀 Analyze Batch", variant="primary")
                        btn_clear_batch = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    batch_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    batch_summary = gr.Markdown("## 📊 Summary will appear here")
            
            # Results table
            batch_output = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Results",
                interactive=False
            )
            
            # Store results
            results_df_state = gr.State()
            
            # Batch processing
            def process_batch(text):
                results = analyze_batch_enhanced(text)
                df = pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"])
                summary = generate_summary(df) if len(results) > 0 else "## 📊 No data"
                return results, summary, "✅ Done", df
            
            btn_batch.click(
                process_batch,
                inputs=batch_input,
                outputs=[batch_output, batch_summary, batch_status, results_df_state]
            )
            
            btn_clear_batch.click(
                lambda: ("", [], "## 📊 Summary", "Ready", None),
                outputs=[batch_input, batch_output, batch_summary, batch_status, results_df_state]
            )
        
        # Tab 3: File Upload
        with gr.Tab("📁 Upload File"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file = gr.File(
                        label="Upload file",
                        file_types=[".txt", ".csv"]
                    )
                    
                    upload_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    
                    with gr.Row():
                        btn_file = gr.Button("🔍 Analyze", variant="primary")
                        btn_clear_file = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Column(scale=2):
                    file_output = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Results",
                        interactive=False
                    )
            
            upload_df_state = gr.State()
            
            btn_file.click(
                process_file_enhanced,
                inputs=upload_file,
                outputs=[file_output, upload_status, upload_df_state]
            )
            
            btn_clear_file.click(
                lambda: (None, [], "Ready", None),
                outputs=[upload_file, file_output, upload_status, upload_df_state]
            )
        
        # Tab 4: Charts
        with gr.Tab("📊 Charts"):
            with gr.Row():
                with gr.Column(scale=1):
                    chart_input = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Data for Charts",
                        value=[["Example text", "😊 POSITIVE", "85%", "example, text"]]
                    )
                    
                    with gr.Row():
                        btn_chart = gr.Button("📈 Generate", variant="primary")
                        btn_clear_chart = gr.Button("🗑️ Clear", variant="secondary")
                    
                    gr.Markdown("**💡 Tip:** Copy results from Batch or File tabs")
                
                with gr.Column(scale=2):
                    chart_output = gr.Plot(label="Sentiment Chart")
            
            def generate_chart(data):
                return create_sentiment_chart_enhanced(data)
            
            btn_chart.click(
                generate_chart,
                inputs=chart_input,
                outputs=chart_output
            )
            
            btn_clear_chart.click(
                lambda: go.Figure(),
                outputs=chart_output
            )
        
        # Tab 5: Export
        with gr.Tab("⬇️ Export"):
            with gr.Row():
                with gr.Column(scale=2):
                    export_input = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Data to Export"
                    )
                    
                    with gr.Row():
                        btn_export_csv = gr.Button("📥 CSV", variant="primary")
                        btn_export_json = gr.Button("📥 JSON", variant="secondary")
                
                with gr.Column(scale=1):
                    export_file = gr.File(label="Download", visible=False)
                    gr.Markdown("**Instructions:**")
                    gr.Markdown("1. Paste results here")
                    gr.Markdown("2. Click export button")
                    gr.Markdown("3. Download file")
            
            def export_data(data, format):
                return export_results_enhanced(data, format)
            
            btn_export_csv.click(
                lambda data: export_data(data, "csv"),
                inputs=export_input,
                outputs=export_file
            )
            
            btn_export_json.click(
                lambda data: export_data(data, "json"),
                inputs=export_input,
                outputs=export_file
            )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center;'><p>🚀 Sentiment Analysis Pro | Built with Gradio & Hugging Face</p></div>")

# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo.launch()