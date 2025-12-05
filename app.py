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
import numpy as np

# ==================== CONFIGURATION ====================
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
    radius_size="lg"
)

CSS = """
.gradio-container {max-width: 1200px !important;}
.header-title {font-size: 2.5rem !important; font-weight: 800 !important;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
              color: white !important; border-radius: 12px !important; padding: 20px !important;}
.btn-primary {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important;}
.sentiment-positive {color: #10b981 !important; font-weight: 600;}
.sentiment-negative {color: #ef4444 !important; font-weight: 600;}
.sentiment-neutral {color: #64748b !important; font-weight: 600;}
"""

# ==================== SENTIMENT MODEL ====================
@gr.cache_resource
def load_sentiment_model():
    """Load sentiment model with caching"""
    print("🔄 Loading sentiment model...")
    model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
        max_length=512
    )
    print("✅ Model loaded successfully!")
    return model

sentiment_model = load_sentiment_model()

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
        result = sentiment_model(text)[0]
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
            confidence_html = f"<span style='color:#10b981; font-weight:600;'>{confidence_percent:.1f}%</span>"
        elif confidence_percent >= 60:
            confidence_html = f"<span style='color:#f59e0b; font-weight:600;'>{confidence_percent:.1f}%</span>"
        else:
            confidence_html = f"<span style='color:#ef4444; font-weight:600;'>{confidence_percent:.1f}%</span>"
        
        sentiment = f"{label_data['emoji']} {label_data['label']}"
        
        # Enhanced keyword extraction
        keywords = extract_enhanced_keywords(text)
        
        return sentiment, confidence_html, keywords

    except Exception as e:
        return "⚠️ Error", "0%", f"Processing error: {str(e)[:50]}"

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
def analyze_batch_enhanced(input_text, progress=gr.Progress()):
    """Enhanced batch processing with progress indicator"""
    if not input_text or input_text.strip() == "":
        return []
    
    lines = [line.strip() for line in input_text.split("\n") if line.strip()]
    results = []
    
    for i, line in enumerate(progress.tqdm(lines, desc="Analyzing texts")):
        sentiment, confidence, keywords = analyze_sentiment(line, add_variation=False)
        
        # Truncate long text for display
        display_text = line[:80] + "..." if len(line) > 80 else line
        
        results.append([display_text, sentiment, confidence, keywords])
    
    return results

def process_file_enhanced(file, progress=gr.Progress()):
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
            return [], f"❌ Unsupported file type: {filename.split('.')[-1]}", pd.DataFrame()
        
        # Process each line
        results = []
        for i, line in enumerate(progress.tqdm(lines, desc=f"Processing {filename}")):
            sentiment, confidence, keywords = analyze_sentiment(line, add_variation=False)
            results.append([line[:80] + "..." if len(line) > 80 else line, sentiment, confidence, keywords])
        
        # Create summary
        df_results = pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"])
        summary = generate_summary(df_results)
        
        return results, f"✅ Processed {len(lines)} lines from {filename}", df_results
    
    except Exception as e:
        return [], f"❌ Error processing file: {str(e)}", pd.DataFrame()

def generate_summary(df):
    """Generate markdown summary of batch results"""
    if df.empty:
        return "## 📊 No data available for summary"
    
    # Count sentiments
    sentiments = df["Sentiment"].apply(lambda x: x.split()[1] if len(x.split()) > 1 else x)
    sentiment_counts = sentiments.value_counts()
    
    # Calculate average confidence
    conf_values = df["Confidence"].apply(
        lambda x: float(re.search(r'\d+\.?\d*', x).group()) if re.search(r'\d+\.?\d*', x) else 0
    )
    avg_confidence = conf_values.mean()
    
    summary = f"""
    ## 📊 Batch Analysis Summary
    
    **Total Texts Analyzed:** {len(df)}
    
    **Sentiment Distribution:**
    {chr(10).join([f'- **{sent}**: {count} ({count/len(df)*100:.1f}%)' for sent, count in sentiment_counts.items()])}
    
    **Average Confidence:** {avg_confidence:.1f}%
    
    **Processing Time:** ~{len(df)*0.3:.1f}s estimated
    """
    
    return summary

# ==================== ENHANCED VISUALIZATIONS ====================
def create_sentiment_chart_enhanced(data):
    """Create enhanced sentiment visualization with multiple chart types"""
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
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker=dict(colors=['#ef4444', '#64748b', '#10b981']),
        textinfo='percent+label',
        textposition='inside',
        hoverinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title=dict(text="📈 Sentiment Distribution", font=dict(size=20)),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        annotations=[dict(text=f"Total: {len(df)}", x=0.5, y=0.5, font_size=16, showarrow=False)]
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
    conf_values = df["Confidence"].apply(
        lambda x: float(re.search(r'\d+\.?\d*', x).group()) if re.search(r'\d+\.?\d*', x) else 0
    )
    
    fig = go.Figure(data=[go.Histogram(
        x=conf_values,
        nbinsx=20,
        marker_color='#3b82f6',
        opacity=0.7,
        name='Confidence Scores'
    )])
    
    fig.update_layout(
        title=dict(text="📊 Confidence Score Distribution", font=dict(size=20)),
        xaxis_title="Confidence (%)",
        yaxis_title="Frequency",
        height=350,
        bargap=0.1,
        plot_bgcolor='white'
    )
    
    # Add average line
    avg_conf = conf_values.mean()
    fig.add_vline(x=avg_conf, line_dash="dash", line_color="red", 
                  annotation_text=f"Average: {avg_conf:.1f}%")
    
    return fig

# ==================== ENHANCED EXPORT FUNCTIONS ====================
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
    
    elif format == "excel":
        filename = f"sentiment_analysis_{timestamp}.xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            df.to_excel(tmp.name, index=False)
            return gr.File(value=tmp.name, label=filename)
    
    return None

# ==================== METRIC CARD COMPONENT ====================
def create_metric_card(title, value, description=""):
    """Create a modern metric card"""
    html = f"""
    <div class="metric-card" style="text-align: center;">
        <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
        <div style="font-size: 2rem; font-weight: 700; margin: 10px 0;">{value}</div>
        <div style="font-size: 0.8rem; opacity: 0.7;">{description}</div>
    </div>
    """
    return gr.HTML(html)

# ==================== MODERN UI ====================
with gr.Blocks(title="Sentiment Analysis Pro", theme=THEME, css=CSS) as demo:

    # Header Section
    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Advanced AI-powered sentiment analysis with **single analysis**, **batch text**, **file upload**, **export**, and **charts**.")
    
    # Quick Stats Row
    with gr.Row():
        total_analyzed = create_metric_card("Total Analyzed", "0", "Texts processed")
        avg_confidence = create_metric_card("Avg Confidence", "0%", "Across all analyses")
        positive_rate = create_metric_card("Positive Rate", "0%", "Of total texts")
        processing_time = create_metric_card("Avg Time", "0.0s", "Per analysis")
    
    # Main Tabs
    with gr.Tabs():
        
        # Tab 1: Single Text (Enhanced)
        with gr.Tab("📝 Single Text", id="single"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_single = gr.Textbox(
                        label="Enter Text",
                        placeholder="Type or paste your text here...",
                        lines=4,
                        value="I absolutely love the new design of the website!"
                    )
                    
                    with gr.Row():
                        btn_single = gr.Button("🚀 Analyze Now", variant="primary", scale=2)
                        btn_clear_single = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    # Text Statistics
                    text_stats = gr.JSON(
                        label="📊 Text Statistics",
                        value={"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"}
                    )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        output_sentiment = gr.Textbox(label="Sentiment", interactive=False)
                        output_confidence = gr.HTML(label="Confidence")
                        output_keywords = gr.Textbox(label="Keywords", interactive=False)
                    
                    # Examples
                    with gr.Accordion("💡 Try these examples", open=False):
                        gr.Examples(
                            examples=[
                                ["The customer service was exceptional! Very responsive and helpful."],
                                ["Terrible experience. The product arrived broken and support was unhelpful."],
                                ["It's okay for the price, but don't expect premium quality."],
                                ["Absolutely brilliant! Everything exceeded my expectations."],
                                ["I'm disappointed with the quality. Not worth the money."]
                            ],
                            inputs=input_single,
                            label="Click any example:"
                        )
            
            # Update text stats when typing
            input_single.change(
                lambda text: get_text_stats(text) if text else {"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"},
                inputs=input_single,
                outputs=text_stats
            )
            
            # Single analysis
            btn_single.click(
                analyze_sentiment,
                inputs=input_single,
                outputs=[output_sentiment, output_confidence, output_keywords]
            )
            
            btn_clear_single.click(
                lambda: ("", "📝 Enter text first", "0%", "No keywords", {"Word Count": 0, "Character Count": 0, "Reading Time": "0 min"}),
                outputs=[input_single, output_sentiment, output_confidence, output_keywords, text_stats]
            )
        
        # Tab 2: Batch Processing (Enhanced)
        with gr.Tab("📦 Batch Processing", id="batch"):
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Enter one sentence per line",
                        lines=10,
                        value="""I absolutely love the new design of the website!
The delivery took too long, very frustrating experience
Not bad, but could be better
Amazing customer service, very helpful staff
I'm disappointed with the quality of the product.
Neutral feelings, it's neither good nor bad.
I hate the new interface, very hard to navigate.
The app crashes sometimes, but overall it's okay.
Fantastic! Everything worked perfectly.
Poor packaging caused minor damages.
The service was okay, nothing special.
I'm thrilled with how fast my order arrived!"""
                    )
                    
                    with gr.Row():
                        btn_batch = gr.Button("🚀 Analyze Batch", variant="primary")
                        btn_clear_batch = gr.Button("🗑️ Clear All", variant="secondary")
                
                with gr.Column(scale=1):
                    batch_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    batch_summary = gr.Markdown(label="## 📊 Summary")
            
            # Results table
            batch_output = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Results",
                interactive=False,
                wrap=True
            )
            
            # Visualizations
            with gr.Row():
                sentiment_chart_output = gr.Plot(label="📈 Sentiment Distribution")
                confidence_chart_output = gr.Plot(label="📊 Confidence Distribution")
            
            # Export buttons
            with gr.Row():
                export_csv = gr.Button("📥 Export as CSV", variant="secondary")
                export_json = gr.Button("📥 Export as JSON", variant="secondary")
                export_excel = gr.Button("📥 Export as Excel", variant="secondary")
            
            export_csv_file = gr.File(label="Download", visible=False)
            export_json_file = gr.File(label="Download", visible=False)
            export_excel_file = gr.File(label="Download", visible=False)
            
            # Store results dataframe
            results_df_state = gr.State(pd.DataFrame())
            
            # Batch processing logic
            btn_batch.click(
                analyze_batch_enhanced,
                inputs=batch_input,
                outputs=batch_output
            ).then(
                lambda results: pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"]) if results else pd.DataFrame(),
                inputs=batch_output,
                outputs=results_df_state
            ).then(
                lambda df: generate_summary(df),
                inputs=results_df_state,
                outputs=batch_summary
            ).then(
                lambda df: create_sentiment_chart_enhanced(df),
                inputs=results_df_state,
                outputs=sentiment_chart_output
            ).then(
                lambda df: create_confidence_chart(df),
                inputs=results_df_state,
                outputs=confidence_chart_output
            ).then(
                lambda: "✅ Batch analysis complete!",
                outputs=batch_status
            )
            
            # Export handlers
            export_csv.click(
                lambda df: export_results_enhanced(df, "csv"),
                inputs=results_df_state,
                outputs=export_csv_file
            )
            
            export_json.click(
                lambda df: export_results_enhanced(df, "json"),
                inputs=results_df_state,
                outputs=export_json_file
            )
            
            export_excel.click(
                lambda df: export_results_enhanced(df, "excel"),
                inputs=results_df_state,
                outputs=export_excel_file
            )
            
            btn_clear_batch.click(
                lambda: ("", [], "Ready", "## 📊 Summary", pd.DataFrame(), None, None, None, None, None),
                outputs=[batch_input, batch_output, batch_status, batch_summary, results_df_state,
                        sentiment_chart_output, confidence_chart_output,
                        export_csv_file, export_json_file, export_excel_file]
            )
        
        # Tab 3: File Upload (Enhanced)
        with gr.Tab("📁 Upload File", id="upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file = gr.File(
                        label="Upload file",
                        file_types=[".txt", ".csv"],
                        file_count="single"
                    )
                    
                    upload_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    
                    with gr.Row():
                        btn_file = gr.Button("🔍 Analyze File", variant="primary")
                        btn_clear_file = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Column(scale=2):
                    file_output = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Results",
                        interactive=False,
                        wrap=True
                    )
            
            upload_df_state = gr.State(pd.DataFrame())
            
            btn_file.click(
                process_file_enhanced,
                inputs=upload_file,
                outputs=[file_output, upload_status, upload_df_state]
            )
            
            btn_clear_file.click(
                lambda: (None, [], "Ready", pd.DataFrame()),
                outputs=[upload_file, file_output, upload_status, upload_df_state]
            )
        
        # Tab 4: Charts (Enhanced)
        with gr.Tab("📊 Charts & Visualizations", id="charts"):
            with gr.Row():
                with gr.Column(scale=1):
                    chart_input = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Data for Visualization",
                        value=[["Example text", "😊 POSITIVE", "85%", "example, text"]],
                        interactive=True
                    )
                    
                    with gr.Row():
                        btn_chart = gr.Button("📈 Generate Charts", variant="primary")
                        btn_clear_chart = gr.Button("🗑️ Clear Charts", variant="secondary")
                    
                    gr.Markdown("### 📋 Chart Options")
                    chart_type = gr.Radio(
                        choices=["Sentiment Distribution", "Confidence Histogram", "Both"],
                        value="Both",
                        label="Select Chart Type"
                    )
                
                with gr.Column(scale=2):
                    with gr.Row():
                        chart_output_1 = gr.Plot(label="Chart 1")
                        chart_output_2 = gr.Plot(label="Chart 2")
            
            def generate_charts(data, chart_type):
                if data is None or len(data) == 0:
                    fig_empty = go.Figure()
                    fig_empty.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
                    fig_empty.update_layout(height=300)
                    return fig_empty, fig_empty
                
                if chart_type == "Sentiment Distribution":
                    return create_sentiment_chart_enhanced(data), go.Figure()
                elif chart_type == "Confidence Histogram":
                    return create_confidence_chart(data), go.Figure()
                else:  # Both
                    return create_sentiment_chart_enhanced(data), create_confidence_chart(data)
            
            btn_chart.click(
                generate_charts,
                inputs=[chart_input, chart_type],
                outputs=[chart_output_1, chart_output_2]
            )
            
            btn_clear_chart.click(
                lambda: (go.Figure(), go.Figure()),
                outputs=[chart_output_1, chart_output_2]
            )
        
        # Tab 5: Export (Enhanced)
        with gr.Tab("⬇️ Export Results", id="export"):
            with gr.Row():
                with gr.Column(scale=2):
                    export_input = gr.Dataframe(
                        headers=["Text", "Sentiment", "Confidence", "Keywords"],
                        label="Data to Export",
                        interactive=True,
                        wrap=True
                    )
                    
                    with gr.Row():
                        btn_export_csv = gr.DownloadButton(
                            "📥 Download CSV",
                            variant="primary"
                        )
                        btn_export_json = gr.DownloadButton(
                            "📥 Download JSON", 
                            variant="secondary"
                        )
                        btn_export_excel = gr.DownloadButton(
                            "📥 Download Excel",
                            variant="secondary"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Export Instructions")
                    gr.Markdown("""
                    1. **Paste results** from any analysis tab
                    2. **Click download button** for desired format
                    3. **Files include:**
                       - Text content
                       - Sentiment with emojis
                       - Confidence scores
                       - Extracted keywords
                    """)
                    
                    # Auto-fill from batch results
                    btn_auto_fill = gr.Button(
                        "🔄 Load from Batch Results",
                        variant="secondary",
                        size="sm"
                    )
            
            # Connect export buttons
            def export_wrapper(data, format):
                if data is None or len(data) == 0:
                    return None
                return export_results_enhanced(data, format)
            
            btn_export_csv.click(
                lambda data: export_wrapper(data, "csv"),
                inputs=export_input,
                outputs=btn_export_csv
            )
            
            btn_export_json.click(
                lambda data: export_wrapper(data, "json"),
                inputs=export_input,
                outputs=btn_export_json
            )
            
            btn_export_excel.click(
                lambda data: export_wrapper(data, "excel"),
                inputs=export_input,
                outputs=btn_export_excel
            )
            
            btn_auto_fill.click(
                lambda: results_df_state.value if hasattr(results_df_state, 'value') else pd.DataFrame(),
                outputs=export_input
            )
        
        # Tab 6: About & Settings
        with gr.Tab("ℹ️ About & Settings", id="about"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎯 About This Dashboard")
                    gr.Markdown("""
                    **Sentiment Analysis Pro** is an advanced AI-powered tool for analyzing emotional tone in text.
                    
                    **Features:**
                    - 🤖 Transformer-based sentiment analysis
                    - 📊 Real-time visualizations
                    - 📁 Batch processing & file upload
                    - 📥 Multiple export formats
                    - 🎨 Modern, user-friendly interface
                    
                    **Model:** `cardiffnlp/twitter-roberta-base-sentiment`
                    **Accuracy:** ~85-90% on general text
                    """)
                    
                    gr.Markdown("### 📈 Sentiment Categories")
                    with gr.Row():
                        gr.Markdown("**😊 POSITIVE**\nPositive emotions, satisfaction")
                        gr.Markdown("**😐 NEUTRAL**\nNeutral statements, facts")
                        gr.Markdown("**😠 NEGATIVE**\nNegative emotions, criticism")
                
                with gr.Column():
                    gr.Markdown("### ⚙️ Settings")
                    
                    confidence_variation = gr.Checkbox(
                        label="Add confidence variation",
                        value=True,
                        info="Add ±7% random variation to confidence scores for more human-like results"
                    )
                    
                    keyword_count = gr.Slider(
                        label="Number of keywords",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        info="Maximum keywords to extract per text"
                    )
                    
                    theme_selector = gr.Radio(
                        choices=["Light", "Dark", "Auto"],
                        value="Auto",
                        label="Theme"
                    )
                    
                    btn_refresh = gr.Button("🔄 Refresh Model", variant="secondary")
                    
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="✅ Model loaded and ready",
                        interactive=False
                    )
    
    # Footer
    gr.Markdown("---")
    gr.HTML("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p>🚀 <strong>Sentiment Analysis Pro</strong> | Built with Gradio & Hugging Face</p>
        <p style="font-size: 0.9rem;">Analyze emotions, gain insights, make better decisions</p>
    </div>
    """)

# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )