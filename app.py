import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Feedback Emotion Analyzer", page_icon="💬", layout="wide")

# ---------- Links (fill the empty ones later) ----------
LINKS = {
    "💻 GitHub Repository": "https://github.com/Hurerah-create/feedback-emotion-analyzer",
    "📄 Project Report": "https://docs.google.com/document/d/1ALrzePsn29Koc6pLbs7yAQu3HrnZEwoMq11Lqi467Mo/edit",
    "📊 Presentation": "https://docs.google.com/presentation/d/1kpteRRXVeDgWR9eKlwvIdESI2DEjLB6V/edit",
    "🎥 YouTube Video": "",
    "💼 LinkedIn Post": "",
    "🗂️ Dataset (GoEmotions)": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
}

EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😄",
         "sadness": "😢", "surprise": "😲", "neutral": "😐"}
GROUP = {"anger": "Negative", "disgust": "Negative", "fear": "Negative", "sadness": "Negative",
         "joy": "Positive", "surprise": "Ambiguous", "neutral": "Neutral"}
COLOR = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6", "Ambiguous": "#f1c40f"}
SENT_ICON = {"Positive": "👍", "Negative": "👎", "Neutral": "😶", "Ambiguous": "🤔"}

EXAMPLES = {
    "😄 Happy customer": "The delivery was super fast and the packaging was great, thank you so much!",
    "😠 Angry customer": "This is the worst service ever. I am so annoyed that nobody replies to my emails.",
    "😢 Disappointed": "I am really disappointed. The app keeps crashing and I lost all my work.",
    "😨 Worried user": "I am worried that my personal data is not safe in this app.",
    "😲 Surprised user": "Wow, I did not expect the new update to change so much!",
    "😐 Neutral note": "I received the order today and I will check it tomorrow.",
}

SAMPLE = pd.DataFrame({"feedback": [
    "Great product, works exactly as described. Very happy with it!",
    "The support team was rude and did not solve my problem.",
    "Delivery arrived on Tuesday.",
    "I love the new design, it is so clean and easy to use.",
    "Terrible quality. It broke after two days and I am furious.",
    "Why did the price suddenly change? That is a surprise.",
    "I am scared that I will lose my account after the update.",
    "Thanks a lot for the quick refund, much appreciated!",
    "The food was cold and tasted disgusting.",
    "It is okay, nothing special.",
    "So sad that you removed my favourite feature.",
    "Amazing service, the staff were kind and helpful!",
]})

# ---------- Style ----------
st.markdown(
    """
<style>
.hero {
    background: linear-gradient(135deg, #0f766e 0%, #2563eb 50%, #7c3aed 100%);
    padding: 2.2rem 2rem; border-radius: 18px; text-align: center;
    margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.hero h1 { color: white; margin: 0; font-size: 2.4rem; }
.hero p { color: #e0e9ff; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
.card {
    background: #161b26; border: 1px solid #2c3446; border-radius: 14px;
    padding: 1.1rem 1.2rem; height: 100%;
}
.card h4 { margin: 0 0 0.4rem 0; color: #8ab4ff; }
.sent {
    text-align: center; font-size: 1.8rem; font-weight: 800; border-radius: 12px;
    padding: 0.8rem 1rem; margin: 0.4rem 0 1rem 0;
}
.footer { text-align: center; color: #8a93a6; margin-top: 2rem; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_model")
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()
    return tok, mdl


with st.spinner("Waking up the AI model..."):
    tok, model = load_model()

LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]


def predict(texts, batch_size=32, progress=None):
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=64,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        chunks.append(torch.softmax(logits, dim=-1).numpy())
        if progress:
            progress(min(i + batch_size, len(texts)) / len(texts))
    return np.vstack(chunks)


def sentiment_of(probs):
    groups = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0, "Ambiguous": 0.0}
    for lab, p in zip(LABELS, probs):
        groups[GROUP[lab]] += float(p)
    best = max(groups, key=groups.get)
    return best, groups


def set_text(t):
    st.session_state["text"] = t


if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    low_conf = st.slider("Low-confidence warning below", 0.30, 0.90, 0.50, 0.05,
                         help="Show a warning when the AI is not sure about its answer.")
    show_all = st.checkbox("Show all 7 emotion bars", value=True)
    st.divider()
    st.subheader("📌 Project")
    st.write("**Model:** ELECTRA-small")
    st.write("**Task:** 7-emotion classification + sentiment")
    st.write("**Dataset:** GoEmotions")
    st.write("**Framework:** Streamlit")
    st.divider()
    st.subheader("🔗 Project Links")
    for name, url in LINKS.items():
        if url:
            st.link_button(name, url, use_container_width=True)
    st.divider()
    st.caption("Deep Learning Semester Project")

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>💬 Feedback Emotion Analyzer</h1>
  <p>Paste any user feedback. The AI finds the emotion and the sentiment in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Analyze Feedback", "📁 Batch Analysis (CSV)", "📊 Model Performance", "ℹ️ About Project"]
)

# ---------- Tab 1: single text ----------
with tab1:
    st.write("**Try an example:**")
    cols = st.columns(3)
    for i, (label, txt) in enumerate(EXAMPLES.items()):
        cols[i % 3].button(label, on_click=set_text, args=(txt,), use_container_width=True)

    text = st.text_area("✍️ Type or paste the feedback here", key="text", height=130,
                        placeholder="Example: The service was slow but the staff were very kind.")
    go = st.button("🔍 Analyze", type="primary")

    if not go:
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="card"><h4>1️⃣ Write</h4>Type or paste a user review, comment or feedback.</div>', unsafe_allow_html=True)
        c2.markdown('<div class="card"><h4>2️⃣ Analyze</h4>The ELECTRA model scores 7 emotions.</div>', unsafe_allow_html=True)
        c3.markdown('<div class="card"><h4>3️⃣ Understand</h4>See the top emotion and whether the feedback is positive or negative.</div>', unsafe_allow_html=True)
    elif not text.strip():
        st.warning("Please type some feedback first.")
    else:
        bar = st.progress(0, text="📥 Reading the text...")
        time.sleep(0.3)
        bar.progress(30, text="🧠 Running the AI model...")
        t0 = time.time()
        probs = predict([text])[0]
        ms = (time.time() - t0) * 1000
        bar.progress(70, text="📊 Calculating emotions and sentiment...")
        time.sleep(0.3)
        top = LABELS[int(np.argmax(probs))]
        conf = float(np.max(probs))
        sent, groups = sentiment_of(probs)
        bar.progress(100, text="✅ Done!")
        time.sleep(0.4)
        bar.empty()

        st.markdown(
            f'<div class="sent" style="background:{COLOR[sent]}22;border:2px solid {COLOR[sent]};'
            f'color:{COLOR[sent]}">{SENT_ICON[sent]} {sent} feedback</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎭 Top emotion", f"{EMOJI[top]} {top.capitalize()}")
        m2.metric("🎯 Confidence", f"{conf * 100:.1f}%")
        m3.metric("📈 Sentiment score", f"{groups[sent] * 100:.1f}%")
        m4.metric("⚡ Time", f"{ms:.0f} ms")

        if conf < low_conf:
            st.warning("The AI is not very sure about this one. The text may mix several emotions.")

        left, right = st.columns(2)
        order = np.argsort(probs)[::-1]
        with left:
            st.subheader("🎭 Emotion scores")
            df = pd.DataFrame({"Probability": probs}, index=[l.capitalize() for l in LABELS])
            st.bar_chart(df)
        with right:
            st.subheader("🏆 Ranking")
            rows = order if show_all else order[:3]
            for idx in rows:
                st.progress(float(probs[idx]),
                            text=f"{EMOJI[LABELS[idx]]} {LABELS[idx].capitalize()}: {probs[idx] * 100:.1f}%")

        st.session_state["history"].insert(0, {
            "Feedback": text[:80] + ("..." if len(text) > 80 else ""),
            "Emotion": f"{EMOJI[top]} {top}",
            "Sentiment": sent,
            "Confidence": f"{conf * 100:.1f}%",
        })
        st.session_state["history"] = st.session_state["history"][:10]

    if st.session_state["history"]:
        st.divider()
        st.subheader("🕘 Recent analyses")
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)

# ---------- Tab 2: batch ----------
with tab2:
    st.subheader("📁 Analyze many feedbacks at once")
    st.write("Upload a CSV file with one column of feedback text, or try the sample data.")
    up = st.file_uploader("📤 Upload a CSV file", type=["csv"])
    data = None
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception:
            st.error("Could not read this file. Please upload a normal CSV file.")
    elif st.toggle("Use sample feedback data"):
        data = SAMPLE

    if data is not None and len(data.columns) > 0:
        col = st.selectbox("Which column has the feedback text?", list(data.columns))
        data = data.dropna(subset=[col]).copy()
        data[col] = data[col].astype(str)
        if len(data) > 2000:
            st.info("Only the first 2,000 rows will be analyzed to keep the app fast.")
            data = data.head(2000)
        st.dataframe(data.head(5), use_container_width=True, hide_index=True)

        if st.button("🚀 Analyze all", type="primary"):
            bar = st.progress(0, text="🧠 Analyzing feedback...")
            probs = predict(data[col].tolist(),
                            progress=lambda f: bar.progress(f, text=f"🧠 Analyzing... {int(f * 100)}%"))
            bar.empty()
            res = pd.DataFrame({
                "feedback": data[col].values,
                "top_emotion": [LABELS[int(np.argmax(p))] for p in probs],
                "sentiment": [sentiment_of(p)[0] for p in probs],
                "confidence": [round(float(np.max(p)), 3) for p in probs],
            })
            st.session_state["batch_result"] = res
            st.balloons()

    res = st.session_state.get("batch_result")
    if res is not None:
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📝 Feedbacks", len(res))
        k2.metric("👍 Positive", f"{(res['sentiment'] == 'Positive').mean() * 100:.0f}%")
        k3.metric("👎 Negative", f"{(res['sentiment'] == 'Negative').mean() * 100:.0f}%")
        k4.metric("🎯 Avg confidence", f"{res['confidence'].mean() * 100:.0f}%")

        a, b = st.columns(2)
        with a:
            st.subheader("Sentiment split")
            st.bar_chart(res["sentiment"].value_counts())
        with b:
            st.subheader("Emotion split")
            st.bar_chart(res["top_emotion"].value_counts())

        st.subheader("🧾 Results")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download results as CSV", data=res.to_csv(index=False).encode("utf-8"),
                           file_name="feedback_results.csv", mime="text/csv")

# ---------- Tab 3: performance ----------
with tab3:
    st.subheader("📊 Test results")
    st.caption("Measured on 4,968 test comments the model never saw during training.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Accuracy", "0.70")
    k2.metric("Macro F1", "0.63")
    k3.metric("Classes", "7")

    perf = pd.DataFrame({
        "Precision": [0.56, 0.51, 0.69, 0.80, 0.54, 0.60, 0.70],
        "Recall": [0.55, 0.46, 0.70, 0.86, 0.61, 0.63, 0.61],
        "F1": [0.55, 0.49, 0.70, 0.83, 0.57, 0.61, 0.65],
        "Test samples": [572, 76, 80, 1863, 283, 488, 1606],
    }, index=["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Neutral"])

    st.subheader("F1 score per emotion")
    st.bar_chart(perf[["F1"]])
    st.dataframe(perf, use_container_width=True)

    st.markdown("**What these mean in easy words**")
    st.markdown(
        "- **Accuracy 0.70:** the AI picks the right emotion for about 7 out of 10 comments.\n"
        "- **Macro F1 0.63:** an average score that treats every emotion equally, even rare ones.\n"
        "- **Joy is easiest** (F1 0.83). **Disgust and anger are hardest** because they look similar to other negative feelings."
    )

# ---------- Tab 4: about ----------
with tab4:
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
<div class="card"><h4>🎯 Problem</h4>
Companies receive thousands of feedback messages. Reading them all is slow.
This app finds the emotion and the sentiment of each message automatically.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>🗂️ Dataset</h4>
GoEmotions: Reddit comments labeled with emotions.
Comments with a single emotion group were kept: 39,555 for training,
4,946 for validation and 4,968 for testing.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>🧩 Emotion groups</h4>
28 original emotions were grouped into 7:<br>
😄 <b>Joy:</b> joy, amusement, approval, excitement, gratitude, love, optimism, pride, admiration, desire, relief, caring<br>
😠 <b>Anger:</b> anger, annoyance, disapproval<br>
😢 <b>Sadness:</b> sadness, disappointment, embarrassment, grief, remorse<br>
😨 <b>Fear:</b> fear, nervousness<br>
😲 <b>Surprise:</b> surprise, realization, confusion, curiosity<br>
🤢 <b>Disgust:</b> disgust<br>
😐 <b>Neutral:</b> neutral</div>
""",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
<div class="card"><h4>🧠 Pipeline</h4>
1. The text is cut into tokens (max 64).<br>
2. <b>ELECTRA-small</b> reads the tokens and scores 7 emotions.<br>
3. The scores are turned into probabilities.<br>
4. Emotions are grouped into <b>Positive, Negative, Neutral or Ambiguous</b> sentiment.<br>
5. <b>Streamlit</b> shows the result.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>🏋️ Training</h4>
Fine-tuned for 4 epochs on a Google Colab T4 GPU. Batch size 64, learning rate 1e-4,
max length 64. About 14 million parameters, so the weights are only about 54 MB.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>⚠️ Limitations</h4>
Trained on Reddit comments, so it can be less accurate on other writing styles.
Sarcasm is hard for the model. Mixed feelings in one message may show a low confidence.
Only English text is supported.</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">Made with ❤️ using ELECTRA, Transformers and Streamlit · Deep Learning Semester Project</div>',
    unsafe_allow_html=True,
)
