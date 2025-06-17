import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page config
st.set_page_config(page_title="Empathic AI Chatbot", page_icon="💬", layout="centered")

st.title("🤖 Empathic AI Chatbot")
st.markdown("This chatbot detects your emotion and responds with empathy using two fine-tuned models.")

# --- Load Emotion Detection Model ---
@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    return pipeline("text-classification", model="abhay2727/emotion_model")

# --- Load Response Generation Model ---
@st.cache_resource(show_spinner="Loading response generation model...")
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("abhay2727/t5small_updated")
    model = AutoModelForSeq2SeqLM.from_pretrained("abhay2727/t5small_updated")
    return tokenizer, model

# Load models
emotion_classifier = load_emotion_model()
response_tokenizer, response_model = load_response_model()

# --- Core function ---
def generate_empathic_response(user_input):
    # Emotion detection
    emotion_result = emotion_classifier(user_input)
    emotion = emotion_result[0]['label']

    # Prompt construction
    prompt = f"Emotion: {emotion} | Input: {user_input}"

    # Generate response
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = response_model.generate(**inputs, max_length=100)
    response = response_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return emotion, response

# --- UI ---
with st.form("chat_form"):
    user_input = st.text_area("📝 Say something...", height=150)
    submitted = st.form_submit_button("Generate Response")

if submitted and user_input.strip():
    emotion, chatbot_reply = generate_empathic_response(user_input)

    st.markdown("### 🎭 Detected Emotion:")
    st.info(emotion)

    st.markdown("### 💬 Empathic Response:")
    st.success(chatbot_reply)
