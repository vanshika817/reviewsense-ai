import json
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
MODEL = "openai/gpt-oss-120b"
MAX_REVIEWS = 100
BATCH = 20

st.set_page_config(page_title="ReviewSense AI", page_icon="💬", layout="wide")


@st.cache_resource
def client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def ask(prompt, json_mode=False):
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        **extra,
    )
    return r.choices[0].message.content


@st.cache_data(show_spinner=False)
def classify(reviews: tuple):
    """Return (sentiment, topic) for each review, in batches."""
    out = []
    for i in range(0, len(reviews), BATCH):
        chunk = reviews[i : i + BATCH]
        numbered = "\n".join(f"{j}. {r[:300]}" for j, r in enumerate(chunk))
        prompt = (
            "Classify each customer review. Sentiment must be Positive, Neutral or Negative. "
            "Topic must be ONE of: Quality, Price, Delivery, Customer Service, Usability, Other.\n"
            'Reply ONLY as JSON: {"results":[{"id":0,"sentiment":"...","topic":"..."}]}\n\n'
            + numbered
        )
        try:
            by_id = {d["id"]: d for d in json.loads(ask(prompt, True))["results"]}
        except Exception:
            by_id = {}  # fall back to defaults if the model returns bad JSON
        for j in range(len(chunk)):
            d = by_id.get(j, {})
            out.append((d.get("sentiment", "Neutral"), d.get("topic", "Other")))
    return out


@st.cache_data(show_spinner=False)
def insights(summary: str):
    return ask(
        "You are a business analyst. Based on this review data, write: "
        "1) Top 3 things customers love, 2) Top 3 complaints, "
        "3) 3 specific actions the owner should take. Plain English, concise.\n\n" + summary
    )


st.title("💬 ReviewSense AI")
st.caption("Upload customer reviews → get sentiment, themes and action items.")

file = st.file_uploader("Upload a CSV of reviews", type="csv")
if not file:
    st.info("Need test data? Use `sample_reviews.csv` from the repo.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
if not text_cols:
    st.error("No text column found in this file.")
    st.stop()

guess = next((c for c in text_cols if "review" in c.lower() or "text" in c.lower()), text_cols[0])
col = st.selectbox("Which column has the review text?", text_cols, index=text_cols.index(guess))
df = df.dropna(subset=[col]).head(MAX_REVIEWS).reset_index(drop=True)
st.write(f"Analyzing **{len(df)}** reviews (max {MAX_REVIEWS}).")

if st.button("Analyze", type="primary"):
    with st.spinner("AI is reading your reviews..."):
        labels = classify(tuple(df[col].astype(str)))
        df["Sentiment"], df["Topic"] = zip(*labels)
        counts = df["Sentiment"].value_counts().to_dict()
        negatives = df[df.Sentiment == "Negative"][col].head(15).tolist()
        positives = df[df.Sentiment == "Positive"][col].head(15).tolist()
        summary = f"Sentiment counts: {counts}\nNegative samples: {negatives}\nPositive samples: {positives}"
        report = insights(summary)

    c1, c2 = st.columns(2)
    colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
    c1.plotly_chart(
        px.pie(df, names="Sentiment", title="Sentiment split", color="Sentiment", color_discrete_map=colors),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.histogram(df, x="Topic", color="Sentiment", title="Topics by sentiment", color_discrete_map=colors),
        use_container_width=True,
    )

    st.subheader("🧠 AI Insights")
    st.markdown(report)

    st.subheader("Labeled data")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download results CSV", df.to_csv(index=False), "reviews_analyzed.csv")
