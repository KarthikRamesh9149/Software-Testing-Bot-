# Software Testing FAQ Bot  

The **Software Testing FAQ Bot** is an AI-powered FAQ chatbot designed to provide precise and context-aware answers to software testing queries. It leverages **NLP, semantic search, and machine learning models** to understand user questions and retrieve the most relevant responses. By integrating **predefined FAQs, real-time intent recognition, and text similarity analysis**, this chatbot enhances user experience by offering structured and dynamic responses.  

The bot supports **both manual question entry and preloaded FAQs**, ensuring flexibility in query handling. It features **custom UI enhancements**, **interactive responses**, and **image-based explanations** where applicable. Optimized for **efficiency and scalability**, it uses **cached embeddings, transformer-based models, and an optimized search algorithm** for fast and relevant responses.  

---

## Features  

- **Natural Language Processing (NLP) for Question Understanding** – Uses **Sentence Transformers** for similarity-based question matching.  
- **Real-Time Intent Recognition** – Distinguishes between greetings and FAQ queries for personalized interaction.  
- **FAQ-Based Query Handling** – Retrieves answers from a structured dataset of software testing questions.  
- **Dynamic Answer Customization** – Allows users to choose between **Brief, Detailed, or Combined** responses.  
- **Interactive Question Suggestions** – Provides suggested questions based on similarity analysis for improved user experience.  
- **Image-Based Explanations** – Displays relevant images for certain answers when available.  
- **Enhanced UI with Custom CSS** – Includes **interactive buttons, chat-like responses, and responsive design** for a better user interface.  

---

## Tech Stack  

- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Natural Language Processing:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Text Generation Model:** DistilGPT-2  
- **Intent Recognition Model:** DistilBERT (`bhadresh-savani/distilbert-base-uncased-emotion`)  
- **Sentiment Analysis Model:** DistilBERT (`distilbert-base-uncased`)  
- **UI & Styling:** Streamlit + Custom CSS  
- **Data Storage:** JSON-based FAQ dataset  

---
