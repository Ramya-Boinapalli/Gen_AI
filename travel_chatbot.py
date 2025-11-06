import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import pyttsx3
import base64
import requests

# --- Page Config (must be first) ---
st.set_page_config(page_title="AI Travel Chatbot ✈️", page_icon="🌍", layout="centered")

# --- Function: Set Background Image ---
def set_background(url):
    response = requests.get(url)
    if response.status_code == 200:
        img_base64 = base64.b64encode(response.content).decode()
        bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(bg_style, unsafe_allow_html=True)

# --- Apply Your Background Image ---
set_background("https://tse4.mm.bing.net/th/id/OIP._OEAoOBU71bcrUxwGRPWAwHaE7?pid=Api&P=0&h=180")

# --- Title & Intro ---
st.title("✈️ AI Travel Chatbot")
st.write("Your intelligent travel companion powered by **RNN, BERT, DialoGPT & SpeechRecognition** 🎙")

# --- Predefined Q&A ---
travel_qna = {
    "how can i book a flight": "You can book a flight by telling me your source, destination, and travel date.",
    "what are the popular destinations": "Popular destinations include Goa, Delhi, Mumbai, Kerala, and Jaipur!",
    "how to cancel my ticket": "To cancel, please visit your bookings section or contact our 24/7 support.",
    "can you suggest hotels": "Sure! We offer budget and premium hotels across all major cities.",
    "how can i rent a car": "You can rent a car by specifying your city and date, and we’ll show you available options.",
    "what are your support hours": "We’re available 24/7 for all travel-related assistance.",
    "do you offer international flights": "Yes! We cover flights to 50+ countries including the USA, UK, Singapore, and Dubai.",
    "what is the refund policy": "Refunds are processed within 5–7 business days depending on your payment method."
}

# --- Function to match predefined Q&A ---
def check_predefined_answer(query):
    query_lower = query.lower()
    for question, answer in travel_qna.items():
        if question in query_lower:
            return answer
    return None

# --- Load AI Models ---
st.info("Loading AI models... please wait ⏳")
bert = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
engine = pyttsx3.init()

# --- User Input ---
st.subheader("🎙 Ask me anything about your travel!")
query = ""

if st.button("🎤 Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.write("You said:", query)
        except Exception:
            st.error("Sorry, I couldn’t understand. Please try again.")
else:
    query = st.text_input("💬 Type your travel query here:")

# --- Process Query ---
if query:
    with st.spinner("Understanding your query..."):
        predefined_answer = check_predefined_answer(query)
        if predefined_answer:
            st.subheader("🤖 Chatbot Reply:")
            st.info(predefined_answer)
            try:
                engine.stop()
                engine.say(predefined_answer)
                engine.runAndWait()
            except RuntimeError:
                pass
        else:
            context = "We offer flights, hotels, and car rentals across major cities."
            result = bert(question=query, context=context)
            intent = result['answer']

            inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
            reply_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

            st.subheader("🧠 Detected Intent:")
            st.success(intent)
            st.subheader("🤖 Chatbot Reply:")
            st.info(reply)

            try:
                engine.stop()
                engine.say(reply)
                engine.runAndWait()
            except RuntimeError:
                pass
