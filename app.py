import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import os
import random

st.set_page_config(
    page_title="Spam Detector: Ultimate AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;}
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        vec = joblib.load('vectorizer.pkl')
        mod = joblib.load('model.pkl')
        return vec, mod
    except Exception as e:
        return None, None

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

vectorizer, model = load_resources()
stop_words, lemmatizer = setup_nltk()

def preprocessing_pipeline(text):
    if not isinstance(text, str): return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

if 'email_content' not in st.session_state:
    st.session_state.email_content = ""

def set_text(text):
    st.session_state.email_content = text

menu = ["🏠 Strona Główna", "🧠 Klasyfikator", "📊 Analiza Modelu"]
choice = st.sidebar.radio("Menu", menu)
st.sidebar.markdown("---")
st.sidebar.info("Autor: Wiktor Pieprzowski")
st.sidebar.info("Indeks: 155657")
st.sidebar.info("Przedmiot: Sztuczna Inteligencja")
st.sidebar.info("Kierunek: Informatyka")
st.sidebar.info("Semestr: 7")
st.sidebar.info("Rok akademicki: 2025/2026")
st.sidebar.info("Tryb: Dzienny")

if choice == "🏠 Strona Główna":
    st.title("🤖 Klasyfikator spam/ham")
    st.markdown("### Projekt Zaliczeniowy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("ℹ️ **Cel Projektu**")
        st.markdown("""
        Projekt realizuje zadanie **binarnej klasyfikacji tekstu** w celu automatycznej detekcji zagrożeń (Spam/Phishing).
        
        System wykorzystuje zaawansowane przetwarzanie języka naturalnego (**NLP**):
        * **Preprocessing:** Czyszczenie szumu, lematyzacja, usuwanie stop-words.
        * **Wektoryzacja:** TF-IDF (Term Frequency-Inverse Document Frequency).
        * **Silnik ML:** Model wytrenowany na zbiorze **5796 wiadomości**, uczący się semantycznych wzorców oszustw.
        """)
        
        st.success("📈 **Dataset Upgrade**")
        st.write("""
        Model został wytrenowany na **zróżnicowanym zbiorze danych** zawierającym zarówno
        spam konsumencki (fałszywe loterie i reklamy), jak i ataki phishingowe.
        Dzięki temu radzi sobie znacznie lepiej niż modele oparte wyłącznie na korpusie Enron.
        """)

    with col2:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjEx/3oKIPnAiaMCws8nOsE/giphy.gif", caption="AI vs Spam")


elif choice == "🧠 Klasyfikator":
    st.title("🧠 Skaner Wiadomości")
    
    if model is None:
        st.error("Ludzie, tu niczego nie ma.")
    else:
        col_input, col_result = st.columns([2, 1])

        with col_input:
            st.subheader("Wprowadź tekst lub wylosuj")
            
            spam_pool = [
                "CONGRATULATIONS! You have been selected as a winner of $1,000,000. CLICK HERE to claim your prize now! No catch.",
                "URGENT: Your bank account has been locked. Please verify your identity immediately to restore access. Click this link.",
                "HOT singles in your area are waiting for you! Sign up for FREE tonight. No credit card required.",
                "Buy VIAGRA generic online, cheap price, guaranteed satisfaction. Fast shipping worldwide.",
                "Get rich quick! Crypto investment opportunity of a lifetime. Double your money in 24 hours. Guaranteed returns."
                "THANK YOU FOR YOUR ORDER. Your subscription to Premium Cloud Services has been auto-renewed for another year. A charge of $499.99 will be deducted from your credit card today. If you did not authorize this purchase, please click the attachment to cancel the subscription and request a full refund immediately.",
                "SECURITY ALERT: We detected an unauthorized login attempt on your bank account from an unrecognized device in Russia. For your safety, your access has been temporarily restricted. Please click the link below to verify your identity and restore your account immediately. Failure to act within 24 hours will result in permanent account suspension.",
                "OFFICIAL NOTIFICATION: We are pleased to inform you that your email address has been selected as the grand winner of the International Global Lottery. You have won a cash prize of $2,500,000. To claim your prize, please reply to this email with your full name and banking details. This offer is valid for a limited time only.",
                "URGENT REQUEST: I am currently in a meeting and cannot take calls. I need you to process an urgent wire transfer to a new vendor immediately. Please reply to this email so I can send you the invoice and banking details. This payment must be processed before the end of the day. Treat this with high priority.",
                "Do you suffer from low energy or performance issues? Our new clinically proven formula guarantees satisfaction and improved health. Buy generic supplements online at a cheap price. No prescription needed. Fast shipping worldwide. Click here to browse our catalog and get a 50% discount today."
            ]
            
            ham_pool = [
                "The meeting has been rescheduled to Monday morning. Please review the attached agenda beforehand.",
                "Vince, I have attached the contract for your review. Let me know if everything looks correct.",
                "Are we still on for lunch today? I have the documents you asked for.",
                "Please find the spreadsheet attached. I finished the analysis yesterday.",
                "Going to the office tomorrow. Do you need anything from the archives?",
                "The strategy meeting has been rescheduled to Monday morning in the Houston office. Please review the attached agenda and prepare your reports beforehand.",
                "Vince, I have attached the gas transportation contract for your final review. Please let me know if the figures look correct before we pass it to legal.",
                "Are we still on for lunch today to discuss the new project? I have the documents you asked for and I printed the analysis.",
                "Kindly find the spreadsheet attached regarding the risk management assessment. I finished the analysis yesterday as requested by the board.",
                "I am going to the main office tomorrow to check the archives. Do you need any specific files or contracts brought back for the meeting?",
            ]

            col_rand1, col_rand2 = st.columns(2)
            
            with col_rand1:
                if st.button("🎲 Generuj SPAM"):
                    st.session_state.email_content = random.choice(spam_pool)
            
            with col_rand2:
                if st.button("🎲 Generuj HAM"):
                    st.session_state.email_content = random.choice(ham_pool)

            user_input = st.text_area("Treść:", value=st.session_state.email_content, height=200)
            
            if st.button("SKANUJ", type="primary"):
                if user_input:
                    clean_text = preprocessing_pipeline(user_input)
                    vec_input = vectorizer.transform([clean_text]).toarray()

                    try:
                        proba = model.predict_proba(vec_input)[0]
                        spam_prob = proba[1]
                        ham_prob = proba[0]
                    except AttributeError:
                        st.error("Model musi mieć probability=True!")
                        st.stop()
                    
                    is_spam = spam_prob > 0.5

                    with col_result:
                        st.markdown("### Wynik Analizy")
                        if is_spam:
                            st.error("🚨 **SPAM WYKRYTY**")
                            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjRiYTdtZnplajB1bjAxenFhbWJjdGg3Nnk3OHFlZGx4a2UzOTMzaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Hae1NrAQWyKA/giphy.gif")
                        else:
                            st.success("✅ **BEZPIECZNA (HAM)**")
                            st.image("https://media.giphy.com/media/111ebonMs90YLu/giphy.gif")
                        
                        st.markdown("---")
                        st.metric("Pewność modelu", f"{max(spam_prob, ham_prob)*100:.1f}%")

                        chart_data = pd.DataFrame({
                            "Ham": [ham_prob],
                            "Spam": [spam_prob]
                        })
                        st.bar_chart(chart_data, color=["#66b3ff", "#ff4d4d"])
                        
                        with st.expander("🔍 Zobacz co widzi model"):
                            st.text("Tekst po czyszczeniu:")
                            st.caption(clean_text)
                else:
                    st.warning("BRAK TEKSTU!")

elif choice == "📊 Analiza Modelu":
    st.title("📊 Wyniki Treningu")
    
    tab1, tab2, tab3 = st.tabs(["📉 Macierz Pomyłek", "☁️ Analiza NLP", "🏆 Modele"])
    
    with tab1:
        st.subheader("Macierz pomyłek")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption='Skuteczność na zbiorze testowym')
        else:
            st.warning("Brak pliku confusion_matrix.png")

    with tab2:
        st.header("🔍 Analiza Lingwistyczna i Wzorce Semantyczne")
        st.markdown("""
        Poniższa sekcja wizualizuje, w jaki sposób model klasyfikuje tekst. Dzięki technikom NLP (Natural Language Processing)
        możemy wyodrębnić słowa kluczowe, które najsilniej różnicują wiadomości bezpieczne od złośliwych.
        """)
        
        st.markdown("### ☁️ Cloud of Words")
        st.write("Porównanie najczęściej występujących słów w obu klasach. Zauważalna jest wyraźna różnica w tonie i tematyce.")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("**HAM**")
            st.caption("Specyficzna mieszanka komunikacji operacyjnej i technicznej. Słowa takie jak `linux`, `user group`, `problem` wskazują na dyskusje specjalistyczne (IT Support/Dev). Reszta to pragmatyczny język pracy: `need`, `work`, `time`, `said`..")
   
        with row1_col2:
            st.markdown("**SPAM**")
            st.caption("Dominacja tagów HTML (`td`, `tr`, `width`, `arial`) w chmurze słów SPAM wynika z faktu, że wiadomości niechciane często zawierają skomplikowany kod formatujący (newslettery, reklamy graficzne), w przeciwieństwie do prostych wiadomości tekstowych w klasie HAM.")
                
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            if os.path.exists('cloudOfWords_HAM.png'): 
                st.image('cloudOfWords_HAM.png', use_container_width=True)
            else:
                st.warning("Brak pliku HAM.")

        with row2_col2:
            if os.path.exists('cloudOfWords_SPAM.png'): 
                st.image('cloudOfWords_SPAM.png', use_container_width=True)
            else:
                st.warning("Brak pliku SPAM.")

        st.markdown("---")
        
        st.subheader("📊 Interpretowalność Modelu (Feature Importance)")
        st.markdown("""Wykres przedstawia **20 słów o największej wadze decyzyjnej**. Są to tokeny, które model uznał za najsilniejsze indykatory spamu.""")
        
        col_feat1, col_feat2 = st.columns([2, 1])
        
        with col_feat1:
            if os.path.exists('top_words_4_spam.png'):
                st.image('top_words_4_spam.png', caption="Wagi cech dla klasy SPAM", use_container_width=True)
            else:
                st.warning("Brak wykresu Feature Importance.")
        
        with col_feat2:
                st.info("💡 **Wnioski Analityczne (SVM)**")
                st.markdown("""
                1. **Agresywne CTA (Call to Action):** Model nadaje gigantyczną wagę czasownikom wymuszającym reakcję: `click` (2.16), `please` (1.98), `remove`, `order`. Spammerzy desperacko walczą o interakcję.
                2. **Finanse i Benefity:** Klasyczne słowa-przynęty: `money`, `free`, `credit`, `fund`, `offer`. To potwierdza, że większość spamu w zbiorze ma podłoże finansowe (scam/reklama).
                3. **Artefakty Techniczne i HTML:** * `facearial`, `wi`, `style` - pozostałości po agresywnym formatowaniu HTML/CSS.
                    * `spamassassin...` - model wykrył sygnatury metadanych specyficzne dla list mailingowych, co (choć jest drogą na skróty) skutecznie identyfikuje źródło zagrożenia.
                """)

    with tab3:
        st.subheader("🏆 Ranking Algorytmów")
        st.write("Zestawienie wyników uzyskanych podczas treningu (automatyczny sync).")
        
        if os.path.exists('model_results.csv'):
            df_results = pd.read_csv('model_results.csv')
            
            df_results = df_results.round(4)
            
            st.dataframe(
                df_results.style.highlight_max(axis=0, subset=['F1-Score', 'Accuracy'], color="#22970d"),
                use_container_width=True,
                hide_index=True
            )
            
            best_model = df_results.iloc[0]['Model']
            best_f1 = df_results.iloc[0]['F1-Score']
            st.success(f"**🥇Najlepszy model:** {best_model} (F1: {best_f1})")
            
        else:
            st.error("Nie znaleziono pliku 'model_results.csv'.")
        
st.markdown("---")
st.markdown("© 2025 Projekt Zaliczeniowy | Wiktor Pieprzowski | indeks: 155657")
