import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import re

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Spam Message Classifier",
    page_icon="📩",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

header[data-testid="stHeader"]{
    background: transparent;
}

.block-container{
    padding-top:0.8rem !important;
}

[data-testid="stSidebar"]{
    background:#0f172a;
}

/* Hero */
.hero{
    background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

/* Text Area Container */
[data-testid="stTextArea"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: #111827 !important;
}

/* Text Area */
[data-testid="stTextArea"] textarea {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: #111827 !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 15px !important;
}

/* Remove focus border */
[data-testid="stTextArea"] textarea:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Remove red border from Streamlit wrapper */
[data-testid="stTextArea"] > div {
    border: none !important;
    box-shadow: none !important;
}

textarea:focus{
    border:none !important;
    outline:none !important;
    box-shadow:none !important;
}

.footer{
    text-align:center;
    color:#9ca3af;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD MODEL ----------------

try:
    with open("models/spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📊 Dashboard")

    st.success("Best Model: SVM (TF-IDF)")
    st.info("Accuracy: 98.06%")

    st.markdown("---")

    st.subheader("📈 Evaluation Metrics")

    st.metric("Accuracy", "98.06%")
    st.metric("Precision", "98.25%")
    st.metric("Recall", "87.50%")
    st.metric("F1 Score", "92.56%")

    st.markdown("---")

    st.subheader("🛠 Tech Stack")

    st.write("Python")
    st.write("Scikit-Learn")
    st.write("NLP")
    st.write("TF-IDF")
    st.write("Streamlit")
    st.write("Plotly")

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
<h1>📩 AI Spam Message Classifier</h1>
<p>Spam Detection using NLP + TF-IDF + SVM</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------

message = st.text_area(
    "Enter Message",
    placeholder="Type your SMS or Email here...",
    height=220,
    label_visibility="collapsed"
)

# ---------------- STATS ----------------

if message:

    chars = len(message)
    words = len(message.split())

    sentences = len(
        [s for s in re.split(r'[.!?]+', message) if s.strip()]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Characters", chars)

    with c2:
        st.metric("Words", words)

    with c3:
        st.metric("Sentences", sentences)

# ---------------- BUTTON ----------------

if st.button("🚀 Analyze Message", width="stretch"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:

        try:

            transformed = vectorizer.transform([message])

            prediction = model.predict(transformed)[0]

            probs = model.predict_proba(transformed)[0]

            not_spam_prob = round(probs[0] * 100, 2)
            spam_prob = round(probs[1] * 100, 2)

            st.divider()

            if prediction == 1:

                st.error(
                    f"🚨 SPAM DETECTED ({spam_prob}%)"
                )

                result = "Spam"

            else:

                st.success(
                    f"✅ NOT SPAM ({not_spam_prob}%)"
                )

                result = "Not Spam"

            # Summary

            st.subheader("📌 Prediction Summary")

            a, b, c = st.columns(3)

            with a:
                st.metric("Result", result)

            with b:
                st.metric("Spam %", f"{spam_prob}%")

            with c:
                st.metric("Not Spam %", f"{not_spam_prob}%")

            # Confidence

            st.subheader("📊 Confidence Analysis")

            left, right = st.columns(2)

            with left:

                st.write("🚨 Spam Probability")

                st.progress(int(spam_prob))

                st.metric(
                    "Spam Score",
                    f"{spam_prob}%"
                )

            with right:

                st.write("✅ Not Spam Probability")

                st.progress(int(not_spam_prob))

                st.metric(
                    "Not Spam Score",
                    f"{not_spam_prob}%"
                )

            # Pie Chart

            chart_df = pd.DataFrame({
                "Category": ["Spam", "Not Spam"],
                "Probability": [spam_prob, not_spam_prob]
            })

            fig = px.pie(
                chart_df,
                names="Category",
                values="Probability",
                hole=0.6,
                title="Prediction Confidence"
            )

            fig.update_layout(
                height=400
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # Save History

            st.session_state.history.append({
                "Message": message[:80],
                "Prediction": result,
                "Spam %": spam_prob,
                "Not Spam %": not_spam_prob
            })

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------- HISTORY ----------------

if st.session_state.history:

    st.divider()

    st.subheader("📝 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        width="stretch"
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        "📥 Download History CSV",
        csv,
        "prediction_history.csv",
        "text/csv"
    )

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div class="footer">

<h3>👨‍💻 Team Members</h3>

Nikita Mishra • Ranjit Bhardwaj • Gaurav Chauhan

<br><br>

Built with ❤️ using Python, NLP, Scikit-Learn & Streamlit

</div>
""", unsafe_allow_html=True)