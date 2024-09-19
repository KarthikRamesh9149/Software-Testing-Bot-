import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib
import warnings

# Suppress warnings and errors from being displayed
warnings.filterwarnings("ignore")

# Set up Streamlit page configuration
st.set_page_config(page_title="Software Testing FAQ Chatbot", page_icon="🟠", layout="centered")

# Custom CSS for improved button responsiveness and consistent answer background styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5DC;
    }
    .chatbox.bot {
        background-color: #FFA500;
        color: #000000;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    .chatbox.user {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #FFA500;
        color: #000000;
        border-radius: 10px;
        padding: 5px 15px;
        border: none;
        font-size: 16px;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
        cursor: pointer;
    }
    .answer-chosen {
        background-color: #FFA500;
        padding: 5px 10px;
        border-radius: 8px;
        color: #000000;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #FFA500;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .typing-indicator {
        color: #000000;
        font-style: italic;
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-weight: 600;
    }
    .suggestion-dropdown {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        max-height: 150px;
        overflow-y: auto;
        position: relative;
        z-index: 9999;
    }
    .suggestion-item {
        padding: 8px;
        cursor: pointer;
    }
    .suggestion-item:hover {
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# Download WordNet data
nltk.download('wordnet')

# Set the dataset and image paths
DATASET_PATH = r'C:\Users\uig83773\Desktop\Software Testing Bot using LLM\FinalDataset.json'
IMAGE_DIR = r'C:\Users\uig83773\Desktop\Software Testing Bot using LLM\images'

# Load JSON FAQ data with caching
@st.cache
def load_faq_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

faq_data = load_faq_data(DATASET_PATH)
questions = [item['question'] for item in faq_data]

# Cache question embeddings to avoid recomputation
@st.cache
def get_question_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions, convert_to_tensor=True)
    return embeddings

question_embeddings = get_question_embeddings()

# Lazy load models
@st.cache
def load_generator():
    return AutoModelForCausalLM.from_pretrained('distilgpt2'), AutoTokenizer.from_pretrained('distilgpt2')

@st.cache
def load_sentiment_analyzer():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased')

@st.cache
def load_intent_recognition_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = ""

if "answer_choice" not in st.session_state:
    st.session_state.answer_choice = "Brief"  # Default answer choice

if "last_question" not in st.session_state:
    st.session_state.last_question = None  # Store the last selected question

# Function to format the answer
def format_answer(brief, detailed):
    formatted_brief = f"**Brief Answer:**\n\n{brief}"
    formatted_detailed = "**Detailed Answer:**\n\n" + "\n".join([f"- {line.strip()}" if line.startswith("-") else line for line in detailed.splitlines() if line.strip()])
    return formatted_brief, formatted_detailed

# Recognize user intent using a basic approach
def recognize_intent(user_input):
    greetings = ["hi", "hello", "hey", "howdy", "greetings"]
    faq_keywords = ["what is", "how do", "describe", "why", "when", "difference between", "how to"]
    
    lower_input = user_input.lower()

    if any(greeting in lower_input for greeting in greetings):
        return "greeting"
    elif any(keyword in lower_input for keyword in faq_keywords):
        return "faq_query"
    return "faq_query"

# Handle greeting responses
def handle_greeting():
    return "Hello! How can I assist you today with your software testing questions?"

# Suggest FAQ questions using cosine similarity
def suggest_questions(user_input):
    user_intent = recognize_intent(user_input)
    if user_intent == "greeting":
        return []
    
    first_words = " ".join(user_input.lower().split()[:4])
    matching_questions = [q for q in questions if q.lower().startswith(first_words)]
    
    if len(matching_questions) < 4:
        user_input_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([user_input], convert_to_tensor=True)
        similarities = cosine_similarity(user_input_embedding, question_embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:5]
        additional_suggestions = [questions[i] for i in top_indices if questions[i] not in matching_questions]
        matching_questions.extend(additional_suggestions)
    
    return matching_questions[:5]

# Function to generate a stable unique key for Streamlit widgets
def generate_unique_key(question_text):
    return hashlib.md5(question_text.encode()).hexdigest()

# Display the selected answer
def display_answer(selected_question, faq_data):
    closest_match = next((item for item in faq_data if item['question'] == selected_question), None)
    if not closest_match:
        st.warning("❌ No close match found.")
        return

    formatted_brief, formatted_detailed = format_answer(closest_match['brief_answer'], closest_match['detailed_answer'])
    
    try:
        # Preserve the user's selection for answer type
        unique_key = generate_unique_key(selected_question)  # Generate stable unique key based on question text
        st.session_state.answer_choice = st.radio("Would you like a brief answer, a detailed answer, or both?", 
                                                  ('Brief', 'Detailed', 'Both'), 
                                                  index=['Brief', 'Detailed', 'Both'].index(st.session_state.answer_choice),
                                                  key=unique_key)  # Using stable key
        
        if st.session_state.answer_choice == 'Brief':
            st.markdown(f"<div class='answer-chosen'>{formatted_brief}</div>", unsafe_allow_html=True)
        elif st.session_state.answer_choice == 'Detailed':
            st.markdown(f"<div class='answer-chosen'>{formatted_detailed}</div>", unsafe_allow_html=True)
        elif st.session_state.answer_choice == 'Both':
            st.markdown(f"<div class='answer-chosen'>{formatted_brief}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-chosen'>{formatted_detailed}</div>", unsafe_allow_html=True)

        display_image(closest_match['image'])
    
    except Exception:
        pass  # Ignore any widget-related warnings

# Display image if it exists
def display_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, use_column_width=True)
    else:
        st.warning(f"⚠️ Image {image_name} not found.")

# Main chatbot interaction
def chatbot():
    st.markdown("<div class='chatbox bot'>Hi! I'm your Software Testing Chatbot.</div>", unsafe_allow_html=True)
    
    choice = st.radio("Would you like to view a list of questions or type your own?", ('View Questions', 'Type Your Own'))

    if choice == 'View Questions':
        st.markdown("### Here are the top 50 questions:")
        selected_question = st.selectbox("Select a question:", [""] + questions[:50])  # Ensure the first option is an empty string
        if selected_question:  # Display only if a question is selected
            display_answer(selected_question, faq_data)

    elif choice == 'Type Your Own':
        user_input = st.text_input("🔍 Start typing your question:")
        if user_input:
            # Check if it's a greeting
            user_intent = recognize_intent(user_input)
            if user_intent == "greeting":
                st.markdown(f"<div class='chatbox bot'>{handle_greeting()}</div>", unsafe_allow_html=True)
            else:
                suggestions = suggest_questions(user_input)
                if suggestions:
                    st.markdown("<div class='suggestion-dropdown'>", unsafe_allow_html=True)
                    for suggestion in suggestions:
                        if st.button(suggestion, key=suggestion):
                            # Update the selected suggestion and reset answer choice
                            st.session_state.selected_suggestion = suggestion
                            st.session_state.answer_choice = "Brief"  # Reset to Brief by default
                            display_answer(suggestion, faq_data)
                    st.markdown("</div>", unsafe_allow_html=True)

        # If a question has been selected, show its answer
        if st.session_state.selected_suggestion:
            display_answer(st.session_state.selected_suggestion, faq_data)

# Main Streamlit UI
def main():
    chatbot()

    if st.checkbox("📜 Show Conversation History"):
        for entry in st.session_state.conversation_history:
            st.markdown(f"<div class='chatbox user'>{entry['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-chosen'>{entry.get('brief_answer')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-chosen'>{entry.get('detailed_answer')}</div>", unsafe_allow_html=True)
            display_image(entry['image'])
        
        st.markdown("<hr><div style='text-align: center;'><small>Powered by Hugging Face & Streamlit</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
