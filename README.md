---
title: Sentiment Dashboard
emoji: 📊
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: Interactive sentiment analysis dashboard
---


# 📊 Sentiment Analysis Dashboard

This interactive dashboard allows users to perform sentiment analysis on text data, including customer reviews, social media posts, or any text content. It provides insights into emotional tone, highlights keywords, visualizes sentiment distributions, and supports export in multiple formats.

## Features

- **Single Text Analysis:** Enter a sentence or paragraph to get sentiment classification, confidence score, and key keywords.
- **Batch Processing:** Analyze multiple lines of text simultaneously.
- **File Upload:** Process `.txt` or `.csv` files containing text data.
- **Visualization:** Generate charts to visualize sentiment distribution.
- **Export Options:** Download results as CSV or JSON.
- **Keyword Extraction:** Highlights the words most influencing sentiment.

## How to Use

1. **Single Text Tab:** Enter your text and click "Analyze" to see sentiment, confidence, and keywords.
2. **Batch Processing Tab:** Enter multiple sentences, one per line, and click "Analyze Batch."
3. **File Upload Tab:** Upload a `.txt` or `.csv` file and click "Analyze File."
4. **Charts Tab:** Paste or use processed results to generate a sentiment distribution chart.
5. **Export Tab:** Download processed data in CSV or JSON format.

## Technical Details

- Built with **Gradio** for an interactive web interface.
- Sentiment analysis powered by Hugging Face Transformers API.
- Supports multi-class classification (Positive / Negative / Neutral).
- Confidence scoring and keyword extraction for deeper insights.
- Proper error handling for invalid inputs and empty text.

## API Limitations (360 words)

While the Hugging Face sentiment analysis models integrated into this dashboard provide robust performance for general text, several limitations should be considered when interpreting results. First, the models may struggle with **sarcasm, irony, or nuanced humor**, as the intended tone often contradicts the literal meaning of the words. For example, the sentence “Great, another delay… just what I needed!” may be misclassified as positive despite its clearly negative sentiment.  

Second, **short texts** or fragments frequently yield lower confidence scores or incorrect classifications due to insufficient context. A single word like “bad” could be negative on its own but may be neutral or positive within a larger phrase, such as “Not bad at all.” Similarly, **mixed sentiments within a sentence** can confuse the model. For instance, “The product is good but the delivery was terrible” might be classified inconsistently depending on which sentiment dominates.  

Third, these models exhibit a **language and cultural bias**. They perform optimally on English-language texts and may misinterpret idioms, slang, or culturally specific references. Emojis, punctuation, and informal language can also unpredictably influence predictions. Confidence scores indicate relative certainty based on the training data, not absolute correctness, and should be interpreted with caution.  

Additionally, **batch processing** has practical limitations. Processing multiple texts simultaneously improves efficiency, but very large datasets may cause memory constraints or slower response times. Keyword extraction highlights frequently used words associated with sentiment but relies on basic frequency or dictionary-based methods, which can overlook nuanced contextual importance.  

Finally, the models are trained on publicly available datasets that may include **biases or outdated language patterns**, potentially affecting accuracy in niche domains, technical contexts, or emerging slang. Users are encouraged to validate results manually for critical applications and to complement model outputs with domain expertise.  

Overall, while these models are highly effective for rapid sentiment insights and exploratory analyses, results should be interpreted cautiously, particularly for complex, nuanced, or sensitive text. Awareness of these limitations ensures informed decision-making when leveraging automated sentiment analysis in real-world applications.

---

## Credits

- Built with **Gradio** and **Hugging Face Transformers API**.
- Developed as part of the "AI in Action" project for the Tech Career Accelerator.
