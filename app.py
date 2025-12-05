import gradio as gr
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import random
import json
from datetime import datetime
import re
from collections import Counter

# ==================== SENTIMENT MODEL ====================
print("🔄 Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512
)
print("✅ Model loaded successfully!")

# Map labels to readable sentiment with emojis
SENTIMENT_MAP = {
    "LABEL_0": {"label": "NEGATIVE", "emoji": "😠"},
    "LABEL_1": {"label": "NEUTRAL", "emoji": "😐"},
    "LABEL_2": {"label": "POSITIVE", "emoji": "😊"}
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "i", "you", "he",
    "she", "it", "we", "they", "this", "that"
}

# ==================== SENTIMENT FUNCTIONS ====================
def analyze_sentiment(text):
    """Analyze sentiment with Hugging Face"""
    if not text or text.strip() == "":
        return "📝 Enter text first", "0%", "No text"

    try:
        result = sentiment_model(text[:512])[0]
        label_data = SENTIMENT_MAP.get(result["label"], SENTIMENT_MAP["LABEL_1"])
        score = result["score"]

        # Add human-like confidence variation
        variation = random.uniform(-0.07, 0.07)
        adjusted_score = max(0, min(1, score + variation))
        confidence = f"{adjusted_score*100:.0f}%"
        
        sentiment = f"{label_data['emoji']} {label_data['label']}"
        
        # Keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [w for w in words if w not in STOPWORDS]
        
        if filtered_words:
            word_counts = Counter(filtered_words)
            common_words = word_counts.most_common(3)
            keywords = ", ".join([word for word, _ in common_words])
        else:
            keywords = "No keywords"
        
        return sentiment, confidence, keywords

    except Exception as e:
        return "⚠️ Error", "0%", f"Error: {str(e)[:50]}"

def analyze_batch(input_text):
    """Process multiple texts"""
    if not input_text or input_text.strip() == "":
        return []
    
    lines = [line.strip() for line in input_text.split("\n") if line.strip()]
    results = []
    
    for line in lines:
        sentiment, confidence, keywords = analyze_sentiment(line)
        display_text = line[:80] + "..." if len(line) > 80 else line
        results.append([display_text, sentiment, confidence, keywords])
    
    return results

def process_file(file):
    """Process uploaded files"""
    if file is None:
        return [], "⚠️ No file uploaded", pd.DataFrame()
    
    try:
        lines = []
        
        if file.name.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
            lines = [l.strip() for l in content.split("\n") if l.strip()]
        
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            if 'text' in df.columns.str.lower().tolist():
                text_col = [col for col in df.columns if 'text' in col.lower()][0]
                lines = df[text_col].astype(str).tolist()
            else:
                lines = df.iloc[:, 0].astype(str).tolist()
        
        else:
            return [], f"❌ Unsupported file type", pd.DataFrame()
        
        # Process each line
        results = []
        for line in lines:
            sentiment, confidence, keywords = analyze_sentiment(line)
            results.append([line[:80] + "..." if len(line) > 80 else line, sentiment, confidence, keywords])
        
        df_results = pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"])
        summary = generate_summary(df_results)
        
        return results, f"✅ Processed {len(lines)} lines", df_results
    
    except Exception as e:
        return [], f"❌ Error: {str(e)}", pd.DataFrame()

def generate_summary(df):
    """Generate summary of results"""
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
    
    summary = f"## 📊 Batch Analysis Summary\n\n"
    summary += f"**Total Texts Analyzed:** {len(df)}\n\n"
    summary += f"**Sentiment Distribution:**\n"
    
    for sent, count in sentiment_counts.items():
        percentage = count/len(df)*100
        summary += f"- **{sent}**: {count} ({percentage:.1f}%)\n"
    
    summary += f"\n**Average Confidence:** {avg_confidence:.1f}%"
    
    return summary

# ==================== VISUALIZATIONS ====================
def create_sentiment_chart(data):
    """Create sentiment chart"""
    if data is None or len(data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="📊 Sentiment Distribution", height=400)
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

# ==================== EXPORT FUNCTIONS ====================
def export_results(data, format="csv"):
    """Export results"""
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

# ==================== MAIN UI ====================
with gr.Blocks(title="Sentiment Analysis Dashboard") as demo:

    # Header
    gr.Markdown("# 📊 Sentiment Analysis Dashboard")
    gr.Markdown("Analyze sentiment in text using AI. Supports single analysis, batch processing, file upload, and charts.")
    
    # Main Tabs
    with gr.Tabs():
        
        # Tab 1: Single Text
        with gr.Tab("📝 Single Text"):
            input_single = gr.Textbox(
                label="Enter Text",
                placeholder="Type your text here...",
                lines=4,
                value="I love the new design of the website!"
            )
            
            btn_single = gr.Button("Analyze")
            
            with gr.Row():
                output_sentiment = gr.Textbox(label="Sentiment")
                output_confidence = gr.Textbox(label="Confidence")
                output_keywords = gr.Textbox(label="Keywords")
            
            btn_single.click(
                analyze_sentiment,
                inputs=input_single,
                outputs=[output_sentiment, output_confidence, output_keywords]
            )
            
            gr.Examples(
                examples=[
                    ["The customer service was exceptional!"],
                    ["Terrible experience with the product."],
                    ["It's okay for the price."],
                ],
                inputs=input_single,
                label="Try these examples:"
            )
        
        # Tab 2: Batch Processing
        with gr.Tab("📦 Batch Processing"):
            batch_input = gr.Textbox(
                label="Enter one sentence per line",
                lines=10,
                value="""I love the new design!
Very frustrating experience.
Could be better.
Excellent customer service.
Disappointed with quality."""
            )
            
            btn_batch = gr.Button("Analyze Batch")
            batch_status = gr.Textbox(label="Status", value="Ready")
            
            batch_output = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Results"
            )
            
            batch_summary = gr.Markdown("## 📊 Summary will appear here")
            
            results_state = gr.State()
            
            def process_batch_wrapper(text):
                results = analyze_batch(text)
                df = pd.DataFrame(results, columns=["Text", "Sentiment", "Confidence", "Keywords"])
                summary = generate_summary(df) if results else "## 📊 No data"
                return results, "✅ Done", summary, df
            
            btn_batch.click(
                process_batch_wrapper,
                inputs=batch_input,
                outputs=[batch_output, batch_status, batch_summary, results_state]
            )
        
        # Tab 3: File Upload
        with gr.Tab("📁 Upload File"):
            upload_file = gr.File(
                label="Upload file (TXT or CSV)",
                file_types=[".txt", ".csv"]
            )
            
            btn_file = gr.Button("Analyze File")
            file_status = gr.Textbox(label="Status", value="Ready")
            
            file_output = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Results"
            )
            
            btn_file.click(
                process_file,
                inputs=upload_file,
                outputs=[file_output, file_status]
            )
        
        # Tab 4: Charts
        with gr.Tab("📊 Charts"):
            chart_input = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Data for Charts",
                value=[["Example text", "😊 POSITIVE", "85%", "example"]]
            )
            
            btn_chart = gr.Button("Generate Chart")
            chart_output = gr.Plot(label="Sentiment Chart")
            
            btn_chart.click(
                create_sentiment_chart,
                inputs=chart_input,
                outputs=chart_output
            )
        
        # Tab 5: Export
        with gr.Tab("⬇️ Export"):
            export_input = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence", "Keywords"],
                label="Data to Export"
            )
            
            with gr.Row():
                btn_export_csv = gr.Button("Export CSV")
                btn_export_json = gr.Button("Export JSON")
            
            export_file = gr.File(label="Download", visible=False)
            
            def export_wrapper(data, format):
                return export_results(data, format)
            
            btn_export_csv.click(
                lambda data: export_wrapper(data, "csv"),
                inputs=export_input,
                outputs=export_file
            )
            
            btn_export_json.click(
                lambda data: export_wrapper(data, "json"),
                inputs=export_input,
                outputs=export_file
            )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center;'>🚀 Sentiment Analysis Dashboard | Built with Gradio & Hugging Face</div>")

# ==================== LAUNCH ====================
if __name__ == "__main__":
    demo.launch()