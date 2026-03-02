# ============================================================
# app.py — Recommandation de Services Hospitaliers
# Modèle: BERT Fine-Tuned
# ============================================================

import streamlit as st
import torch
import numpy as np
import pickle
import os
from transformers import BertTokenizer, BertForSequenceClassification

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="🏥 HospitalAI — Recommandation de Services",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    /* Fond général */
    .main { background-color: #f8f9fa; }
    
    /* Titre principal */
    .main-title {
        text-align: center;
        color: #1a237e;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #546e7a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Carte de résultat */
    .result-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3);
    }
    .result-service {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        color: #90caf9;
        font-size: 1.1rem;
    }
    
    /* Barre de probabilité */
    .prob-bar-container {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.4rem 0;
        border-left: 4px solid #1565c0;
    }
    .prob-label {
        font-weight: 600;
        color: #1a237e;
        font-size: 0.95rem;
    }
    .prob-value {
        color: #1565c0;
        font-weight: 700;
        float: right;
    }
    
    /* Avertissement */
    .warning-box {
        background: #fff8e1;
        border: 1px solid #f9a825;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #e65100;
        margin-top: 1rem;
    }
    
    /* Sidebar */
    .sidebar-info {
        background: #e8eaf6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #283593;
    }
    
    /* Bouton */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def load_model():
    """Chargement du modèle BERT fine-tuned et du tokenizer."""
    model_path = './saved_bert_model'
    
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Charger le label encoder
        with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        num_classes = len(label_encoder.classes_)
        
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes
        )
        model.eval()
        
        return model, tokenizer, label_encoder
    
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None, None, None

def predict(text, model, tokenizer, label_encoder, max_len=128):
    """Effectue une prédiction sur le texte donné."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    predicted_idx = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probs[predicted_idx]
    
    # Toutes les probabilités par classe
    class_probs = {
        label_encoder.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(label_encoder.classes_))
    }
    class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
    
    return predicted_label, confidence, class_probs

# Icônes par service
SERVICE_ICONS = {
    'gynecology': '👶',
    'anesthesia': '💉',
    'TB & Chest disease': '🫁',
    'surgery': '🔬',
    'radiotherapy': '☢️',
    'cardiology': '❤️',
    'oncology': '🎗️',
    'pediatrics': '🧒',
    'neurology': '🧠',
    'orthopedics': '🦴',
    'emergency': '🚨',
    'default': '🏥'
}

def get_icon(service):
    return SERVICE_ICONS.get(service.lower(), SERVICE_ICONS['default'])

# Couleurs pour les barres
def get_color(prob):
    if prob > 0.7:
        return '#1b5e20', '#4caf50'   # Vert foncé, vert
    elif prob > 0.4:
        return '#e65100', '#ff9800'   # Orange foncé, orange
    else:
        return '#b71c1c', '#ef5350'   # Rouge foncé, rouge

# ============================================================
# INTERFACE PRINCIPALE
# ============================================================

# Sidebar
with st.sidebar:
    st.markdown("### ⚕️ À propos")
    st.markdown("""
    <div class='sidebar-info'>
    Application de recommandation de services hospitaliers basée sur l'analyse NLP des avis patients.<br><br>
    <b>Modèle:</b> BERT fine-tuned<br>
    <b>Architecture:</b> Transformer 12 couches<br>
    <b>Pré-entraîné sur:</b> English Wikipedia + Books
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏥 Services détectés")
    model_temp, tok_temp, le_temp = load_model()
    if le_temp is not None:
        for cls in le_temp.classes_:
            icon = get_icon(cls)
            st.markdown(f"{icon} `{cls}`")
    else:
        st.info("Chargez un modèle pour voir les services disponibles.")
    
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    max_len = st.slider("Longueur max tokens", 64, 256, 128, 32)
    show_details = st.checkbox("Afficher détails techniques", False)

# En-tête
st.markdown('<h1 class="main-title">🏥 HospitalAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Recommandation de services hospitaliers par analyse NLP de vos avis patients</p>', unsafe_allow_html=True)

st.markdown("---")

# Chargement modèle
model, tokenizer, label_encoder = load_model()

if model is None:
    st.error("❌ Modèle BERT non trouvé. Veuillez d'abord exécuter le notebook d'entraînement.")
    st.info("Le dossier `saved_bert_model/` doit être présent dans le répertoire courant.")
    
    # Demo mode avec prédictions simulées
    st.warning("🔄 Mode démo activé (sans modèle réel)")
    
    with st.expander("▶ Lancer en mode démo"):
        demo_text = st.text_area(
            "Saisissez votre avis patient:",
            height=120,
            placeholder="Ex: The surgery team was excellent. My procedure went smoothly..."
        )
        if st.button("🔍 Analyser (Démo)"):
            if demo_text.strip():
                demo_classes = ['surgery', 'gynecology', 'anesthesia', 'TB & Chest disease', 'radiotherapy']
                demo_probs = np.random.dirichlet(np.ones(5) * 0.5)
                demo_probs = dict(sorted(zip(demo_classes, demo_probs), key=lambda x: x[1], reverse=True))
                top = list(demo_probs.items())[0]
                
                st.markdown(f"""
                <div class='result-card'>
                    <div class='result-service'>{get_icon(top[0])} {top[0].upper()}</div>
                    <div class='result-confidence'>Confiance: {top[1]*100:.1f}% (DÉMO)</div>
                </div>
                """, unsafe_allow_html=True)
else:
    # Interface principale
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📝 Avis du patient")
    with col2:
        device_info = "🔥 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.markdown(f"<small>{device_info}</small>", unsafe_allow_html=True)
    
    # Zone de texte
    patient_review = st.text_area(
        "",
        height=150,
        placeholder="Saisissez l'avis patient en anglais...\n\nEx: The surgery team was exceptional. My procedure went smoothly and the post-operative care was outstanding...",
        label_visibility="collapsed"
    )
    
    # Exemples rapides
    st.markdown("**Exemples rapides:**")
    col1, col2, col3 = st.columns(3)
    
    example_texts = {
        "🔬 Surgery": "The surgical team performed my appendectomy flawlessly. Recovery was smooth and pain management excellent.",
        "👶 Gynecology": "The gynecology unit was exceptional. The doctor was thorough and empathetic during my prenatal consultation.",
        "🫁 Respiratory": "The pulmonology team expertly managed my breathing difficulties and chest infections. Excellent care."
    }
    
    for col, (label, text) in zip([col1, col2, col3], example_texts.items()):
        with col:
            if st.button(label, use_container_width=True):
                patient_review = text
                st.session_state['example_text'] = text
    
    # Récupérer le texte d'exemple si cliqué
    if 'example_text' in st.session_state and not patient_review:
        patient_review = st.session_state['example_text']
    
    st.markdown("")
    
    # Bouton de prédiction
    predict_btn = st.button("🔍 Analyser l'avis patient", use_container_width=True)
    
    # Prédiction
    if predict_btn and patient_review.strip():
        with st.spinner("🧠 Analyse en cours..."):
            pred_label, confidence, class_probs = predict(
                patient_review, model, tokenizer, label_encoder, max_len
            )
        
        # Résultat principal
        icon = get_icon(pred_label)
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-service'>{icon} {pred_label.upper()}</div>
            <div class='result-confidence'>Confiance: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilités par classe
        st.markdown("#### 📊 Probabilités par service")
        
        for service, prob in class_probs.items():
            dark_color, light_color = get_color(prob)
            svc_icon = get_icon(service)
            bar_width = int(prob * 100)
            
            st.markdown(f"""
            <div class='prob-bar-container' style='border-left-color: {dark_color};'>
                <span class='prob-label'>{svc_icon} {service}</span>
                <span class='prob-value' style='color: {dark_color};'>{prob*100:.1f}%</span>
                <div style='clear: both; margin-top: 6px;'>
                    <div style='background: #e0e0e0; border-radius: 8px; height: 10px;'>
                        <div style='background: linear-gradient(90deg, {dark_color}, {light_color}); 
                                    width: {bar_width}%; height: 10px; border-radius: 8px;
                                    transition: width 0.5s;'></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Avertissement médical
        st.markdown("""
        <div class='warning-box'>
            ⚠️ <b>Avertissement:</b> Cette prédiction est basée sur l'analyse NLP et ne constitue pas un avis médical. 
            Elle est destinée à des fins de triage et d'orientation administrative uniquement.
        </div>
        """, unsafe_allow_html=True)
        
        # Détails techniques
        if show_details:
            with st.expander("🔧 Détails techniques"):
                st.json({
                    "model": "BERT fine-tuned (bert-base-uncased)",
                    "predicted_service": pred_label,
                    "confidence": f"{confidence:.4f}",
                    "device": "CUDA" if torch.cuda.is_available() else "CPU",
                    "max_length": max_len,
                    "num_classes": len(label_encoder.classes_),
                    "all_probabilities": {k: f"{v:.4f}" for k, v in class_probs.items()}
                })
    
    elif predict_btn and not patient_review.strip():
        st.warning("⚠️ Veuillez saisir un avis patient avant d'analyser.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>🏥 HospitalAI • Master 1 IA • Projet NLP • Propulsé par BERT & Streamlit</small></center>",
    unsafe_allow_html=True
)