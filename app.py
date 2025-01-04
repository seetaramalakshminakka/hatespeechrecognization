import streamlit as st  # type: ignore
from model import load_data, train_model
from better_profanity import profanity
from textblob import TextBlob
import time
import nltk

nltk.download('stopwords')
def predict_hate_speech(clf, cv, text):
    processed_text = cv.transform([text])
    prediction = clf.predict(processed_text)

    if hasattr(clf, "decision_function"):
        intensity = abs(clf.decision_function(processed_text)[0]) * 100
    elif hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(processed_text)
        intensity = proba.max() * 100
    else:
        intensity = 50

    return prediction, int(intensity)
def is_negative_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity < 0
def check_profanity_context(text):
    lowered_text = text.lower()
    non_offensive_phrases = [
        "should not", "should avoid", "don't", "must not", "we shouldn't", "avoid", "not", "provoke"
    ]
 
    if any(phrase in lowered_text for phrase in non_offensive_phrases):
        return False 

    if is_negative_sentiment(text):
        return True

    # Check for profanity
    return profanity.contains_profanity(text)

df = load_data('twitter_data.csv')
clf, cv = train_model(df)

profanity.load_censor_words()

st.set_page_config(page_title="Hate Speech Recognition", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.header("Hate Speech Recognition App")
    st.write("Enter a sentence in the text area to check if it contains hate speech.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Designed by **SEETARAMALAKSHMI & TEAM**")

# Streamlit app interface
st.markdown("<h1 class='title'>🛡️ Hate Speech Recognition</h1>", unsafe_allow_html=True)
user_input = st.text_area("Enter text here:", height=150, placeholder="Type your text here...")

def animate_progress_bar(intensity):
    progress_bar = st.progress(0)
    for percent in range(intensity + 1):
        time.sleep(0.01)
        progress_bar.progress(percent)

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = ["No Hate Speech Detected"]

            if check_profanity_context(user_input):
                result = ["Offensive Language Detected"]
                st.markdown("<div class='result-area'>", unsafe_allow_html=True)
                st.markdown("<div class='result-text' style='color: orange;'>⚠️ Result: Offensive Language or Hate Speech Detected</div>", unsafe_allow_html=True)

                intensity = 70
                st.markdown("<h4>Intensity:</h4>", unsafe_allow_html=True)
                animate_progress_bar(intensity)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                result, intensity = predict_hate_speech(clf, cv, user_input)

                st.markdown("<div class='result-area'>", unsafe_allow_html=True)

                if result[0] == "Hate Speech Detected":
                    st.markdown("<div class='result-text' style='color: red;'>🛑 Result: Hate Speech Detected</div>", unsafe_allow_html=True)
                    st.markdown("<h4>Intensity of Hate Speech:</h4>", unsafe_allow_html=True)
                    animate_progress_bar(intensity)
                else:
                    st.markdown("<div class='result-text' style='color: green;'>✅ Result: No Hate and Offensive Speech</div>", unsafe_allow_html=True)
                    intensity = 0
                    st.markdown("<h4>Intensity:</h4>", unsafe_allow_html=True)
                    animate_progress_bar(intensity)

                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter some text to analyze.")
