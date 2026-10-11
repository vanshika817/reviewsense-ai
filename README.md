# 💬 ReviewSense AI

An AI-powered customer review analyzer. Upload a CSV of reviews and get sentiment labels, topic breakdowns, and plain-English action items for business owners.

🔗 **Live Demo:** https://reviewsense-vanshika.streamlit.app

## ✨ What it does
- Upload a CSV file of customer reviews
- AI classifies every review by **sentiment** (Positive / Neutral / Negative) and **topic** (Quality, Price, Delivery, Customer Service, Usability)
- Interactive charts show the sentiment split and topics by sentiment
- AI-generated insights: what customers love, top complaints, and 3 recommended actions
- Download the labeled results as a CSV

## 🛠️ Tech Stack
- **Python**: core language
- **Streamlit**: web interface
- **Pandas**: data processing
- **Plotly**: interactive charts
- **Groq API (GPT-OSS 120B)**: LLM for classification and insights
- **GitHub + Streamlit Community Cloud**: version control and deployment

## 🧠 How it works
1. The CSV is read with Pandas and the review column is auto-detected.
2. Reviews are sent to the LLM in batches of 20 to reduce API calls.
3. The model returns structured JSON (sentiment + topic) for each review.
4. Results are cached with `st.cache_data` so repeated runs don't waste API calls.
5. Aggregated counts and sample reviews are sent to the LLM again to write business insights.

## 🚀 Run locally
```bash
git clone https://github.com/vanshika817/reviewsense-ai.git
cd reviewsense-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` and add your Groq API key:
```toml
GROQ_API_KEY = "your-key-here"
```

Then run:
```bash
streamlit run app.py
```

Use `sample_reviews.csv` to try it out.

## 📚 What I learned
- Integrating a real LLM API into a working data app
- Prompting for structured JSON output and handling bad responses safely
- Batching and caching to control API cost and speed
- Managing secrets properly (never committing API keys)
- Debugging real issues: pandas 3 dtype change, deprecated model names, and Git authentication
- Deploying a live app on Streamlit Cloud

## 🔮 Future improvements
- Add a classical ML baseline (TF-IDF + Logistic Regression) and compare accuracy with the LLM
- Keyword/word cloud for the most common complaints
- Filter results by topic or sentiment
- Support for larger files and multiple languages

## 👩‍💻 Author
**Vanshika Jain**: CS student specializing in Data Science
GitHub: [@vanshika817](https://github.com/vanshika817)
