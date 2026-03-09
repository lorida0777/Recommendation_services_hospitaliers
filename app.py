"""
app.py — Recommandation de Service Hospitalier à partir d'un Avis Patient
Modèle : Logistic Regression + TF-IDF (léger, déployable sur Streamlit Cloud)
"""

import re
import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Téléchargement des ressources NLTK (une seule fois) ────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk()

# ── Chargement des modèles (mis en cache pour performances) ────────────────
@st.cache_resource
def load_models():
    """Charge le vectoriseur TF-IDF, le modèle ML et l'encodeur de labels."""
    try:
        tfidf    = joblib.load('models/tfidf_vectorizer.pkl')
        model    = joblib.load('models/model_ml.pkl')
        encoder  = joblib.load('models/label_encoder.pkl')
        return tfidf, model, encoder
    except FileNotFoundError as e:
        st.error(f"❌ Fichier modèle introuvable : {e}\n"
                 f"Assurez-vous que les fichiers .pkl sont dans le dossier models/")
        st.stop()

tfidf, model, encoder = load_models()

# ── Préprocessing NLP (même pipeline que le notebook) ─────────────────────
@st.cache_resource
def get_nlp_tools():
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, STOP_WORDS = get_nlp_tools()

def clean_text(text: str) -> str:
    """Nettoie un avis patient (pipeline identique au notebook d'entraînement)."""
    text = re.sub(r'&#039;', "'", text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def predict_service(text: str):
    """Prédit le service hospitalier et retourne (service, probabilité, top 3)."""
    cleaned     = clean_text(text)
    vectorized  = tfidf.transform([cleaned])
    proba       = model.predict_proba(vectorized)[0]
    pred_index  = np.argmax(proba)
    pred_label  = encoder.inverse_transform([pred_index])[0]
    confidence  = proba[pred_index]

    # Top 3 services
    top3_idx    = np.argsort(proba)[::-1][:3]
    top3        = [(encoder.inverse_transform([i])[0], proba[i]) for i in top3_idx]

    return pred_label, confidence, top3

# ── Icônes par service ─────────────────────────────────────────────────────
SERVICE_ICONS = {
    'Cardiology'      : '❤️',
    'Dermatology'     : '🩹',
    'Endocrinology'   : '⚖️',
    'Gastroenterology': '🫀',
    'Gynecology'      : '🌸',
    'Neurology'       : '🧠',
    'Oncology'        : '🔬',
    'Psychiatry'      : '🧘',
    'Pulmonology'     : '🫁',
    'Rheumatology'    : '🦴',
    'Urology'         : '💧',
}

# ── Interface Streamlit ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recommandation Hospitalière",
    page_icon="🏥",
    layout="centered"
)

# En-tête
st.title("🏥 Recommandation de Service Hospitalier")
st.markdown(
    "**Entrez un avis ou une description médicale** pour obtenir une recommandation "
    "automatique de service hospitalier basée sur l'analyse NLP."
)
st.divider()

# ── Exemples prédéfinis ────────────────────────────────────────────────────
st.subheader("💡 Exemples rapides")
examples = {
    "Douleurs thoraciques"    : "I have strong chest pains and shortness of breath, my heart is racing.",
    "Problèmes de peau"       : "I have severe acne on my face and back that doesn't respond to any cream.",
    "Anxiété / Dépression"    : "I feel very anxious all the time and have been deeply depressed for months.",
    "Diabète / Poids"         : "I struggle with obesity and my blood sugar is always too high.",
    "Maux de dos"             : "I have chronic back pain and joint stiffness that wakes me up at night.",
    "Infection urinaire"      : "I have a burning sensation when I urinate and need to go very frequently.",
}

col1, col2, col3 = st.columns(3)
selected_example = None
for i, (label, text) in enumerate(examples.items()):
    col = [col1, col2, col3][i % 3]
    if col.button(label, use_container_width=True):
        selected_example = text

# ── Zone de saisie ─────────────────────────────────────────────────────────
st.subheader("📝 Votre avis médical")
default_text = selected_example if selected_example else ""
user_input = st.text_area(
    label="Décrivez vos symptômes ou copiez un avis patient :",
    value=default_text,
    height=130,
    placeholder="Ex : I have been suffering from severe headaches and vision problems for weeks..."
)

# ── Prédiction ─────────────────────────────────────────────────────────────
if st.button("🔍 Analyser", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Veuillez entrer un texte avant d'analyser.")
    elif len(user_input.split()) < 3:
        st.warning("⚠️ Le texte est trop court. Veuillez entrer au moins quelques mots.")
    else:
        with st.spinner("Analyse en cours..."):
            service, confidence, top3 = predict_service(user_input)

        st.divider()
        icon = SERVICE_ICONS.get(service, '🏥')

        # Résultat principal
        st.success(f"### {icon} Service recommandé : **{service}**")

        # Barre de confiance
        confidence_pct = confidence * 100
        color = "green" if confidence_pct >= 70 else "orange" if confidence_pct >= 45 else "red"
        st.markdown(f"**Confiance : {confidence_pct:.1f}%**")
        st.progress(float(confidence))

        # Top 3
        st.subheader("📊 Top 3 des services les plus probables")
        for rank, (svc, prob) in enumerate(top3, 1):
            svc_icon = SERVICE_ICONS.get(svc, '🏥')
            col_label, col_bar, col_pct = st.columns([2, 4, 1])
            col_label.markdown(f"{svc_icon} **{svc}**")
            col_bar.progress(float(prob))
            col_pct.markdown(f"**{prob*100:.1f}%**")

        # Texte nettoyé (optionnel)
        with st.expander("🔍 Voir le texte après préprocessing NLP"):
            st.code(clean_text(user_input), language=None)

        # Avertissement médical
        st.info(
            "⚠️ **Avertissement** : Cette recommandation est générée automatiquement "
            "à des fins académiques et ne remplace pas l'avis d'un professionnel de santé."
        )

# ── Sidebar info ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ À propos du projet")
    st.markdown("""
    **Projet NLP Académique**

    Ce système utilise :
    - **Dataset** : Drug Review (UCI/Kaggle)
    - **Préprocessing** : NLTK (stopwords, lemmatisation)
    - **Vectorisation** : TF-IDF (20 000 features)
    - **Modèle** : Logistic Regression

    ---
    **Services couverts :**
    """)

    for svc, icon in SERVICE_ICONS.items():
        st.markdown(f"{icon} {svc}")

    st.divider()
    st.caption("Développé avec Streamlit · Python · scikit-learn")