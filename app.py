# ═══════════════════════════════════════════════════════════════
#  🏥 HospIA — Recommandation de Services Hospitaliers
#  Application Streamlit avec CamemBERT fine-tuned
#  Auteur : Projet Master 1 IA
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ───────────────────────────────────────────────
# CONFIGURATION GLOBALE DE LA PAGE
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="HospIA — Recommandation Hospitalière",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# CSS PERSONNALISÉ
# ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Police et fond général */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.85; }

    /* Cartes de résultat */
    .result-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f8f9ff 100%);
        border: 2px solid #1565c0;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(21, 101, 192, 0.15);
    }
    .result-card h2 { color: #0d47a1; font-size: 1.8rem; margin: 0; }
    .result-card .service-name {
        font-size: 2rem; font-weight: 800;
        color: #1a237e; margin: 0.4rem 0;
    }
    .result-card .confidence {
        font-size: 1.2rem; color: #1565c0; font-weight: 600;
    }

    /* Badge de sentiment */
    .badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-high   { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
    .badge-medium { background:#fff8e1; color:#f57f17; border:1px solid #ffe082; }
    .badge-low    { background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; }

    /* Barre de probabilité custom */
    .prob-row {
        display: flex; align-items: center;
        margin: 0.5rem 0; gap: 12px;
    }
    .prob-label { min-width: 130px; font-weight: 600; font-size: 0.9rem; color: #1a237e; }
    .prob-bar-bg {
        flex: 1; background: #e9ecef; border-radius: 8px; height: 22px; overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%; border-radius: 8px;
        transition: width 0.8s ease;
        display: flex; align-items: center; padding-left: 8px;
        font-size: 0.78rem; font-weight: 700; color: white;
    }
    .prob-value { min-width: 50px; text-align: right; font-weight: 700;
                  font-size: 0.9rem; color: #333; }

    /* Zone de texte */
    .stTextArea textarea {
        border: 2px solid #bbdefb !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 3px rgba(21,101,192,0.15) !important;
    }

    /* Bouton principal */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(13,71,161,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(13,71,161,0.4) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%) !important;
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* Métriques */
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e3f2fd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-box .val { font-size: 1.6rem; font-weight: 800; color: #0d47a1; }
    .metric-box .lbl { font-size: 0.8rem; color: #666; margin-top: 2px; }

    /* Info box */
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #1565c0;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #1a237e;
    }

    /* Historique */
    .history-item {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #e3f2fd;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# CONSTANTES
# ───────────────────────────────────────────────
SERVICES = [
    "Cardiologie", "Dermatologie", "Gynécologie",
    "Oncologie",   "Orthopédie",   "Pédiatrie", "Urgences"
]

# Icônes par service
ICONS = {
    "Cardiologie":  "❤️",
    "Dermatologie": "🧴",
    "Gynécologie":  "🌸",
    "Oncologie":    "🎗️",
    "Orthopédie":   "🦴",
    "Pédiatrie":    "👶",
    "Urgences":     "🚨",
}

# Couleurs pour les barres de probabilité
COLORS = {
    "Cardiologie":  "#e53935",
    "Dermatologie": "#fb8c00",
    "Gynécologie":  "#e91e8c",
    "Oncologie":    "#8e24aa",
    "Orthopédie":   "#3949ab",
    "Pédiatrie":    "#00acc1",
    "Urgences":     "#f44336",
}

# Descriptions courtes par service
DESCRIPTIONS = {
    "Cardiologie":  "Maladies cardiovasculaires, rythmologie, chirurgie cardiaque.",
    "Dermatologie": "Affections cutanées, dermatoses, cancers de la peau.",
    "Gynécologie":  "Santé féminine, obstétrique, suivi de grossesse.",
    "Oncologie":    "Cancérologie, chimiothérapie, radiothérapie, immunothérapie.",
    "Orthopédie":   "Chirurgie osseuse, articulaire, traumatologie du sport.",
    "Pédiatrie":    "Santé de l'enfant, néonatologie, médecine pédiatrique.",
    "Urgences":     "Prise en charge immédiate, traumatologie, soins d'urgence.",
}

MODEL_NAME = "camembert-base"
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (mis en cache)
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Charge CamemBERT fine-tuné depuis le répertoire local './model'
    ou le modèle de base depuis HuggingFace si le fine-tuné est absent.

    Pour utiliser votre modèle fine-tuné :
        trainer.save_model("./model")
        tokenizer.save_pretrained("./model")
    """
    import os
    model_path = "./model" if os.path.exists("./model") else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(SERVICES),
        id2label={i: s for i, s in enumerate(SERVICES)},
        label2id={s: i for i, s in enumerate(SERVICES)},
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ───────────────────────────────────────────────
# PRÉDICTION
# ───────────────────────────────────────────────
def predict(text: str, tokenizer, model) -> tuple[str, float, dict]:
    """
    Retourne :
        - service prédit (str)
        - confiance (float, 0-1)
        - dict {service: probabilité}
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probas = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probas))
    pred_label = SERVICES[pred_idx]
    confidence = float(probas[pred_idx])
    proba_dict = {svc: float(p) for svc, p in zip(SERVICES, probas)}

    return pred_label, confidence, proba_dict


def nettoyer_texte(texte: str) -> str:
    """Nettoyage léger pour affichage."""
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte


def badge_confiance(conf: float) -> str:
    if conf >= 0.75:
        return '<span class="badge badge-high">✅ Confiance élevée</span>'
    elif conf >= 0.50:
        return '<span class="badge badge-medium">⚠️ Confiance modérée</span>'
    else:
        return '<span class="badge badge-low">❓ Confiance faible</span>'


def barre_probabilite(service: str, proba: float, is_top: bool) -> str:
    """Génère une barre HTML de probabilité."""
    color   = COLORS.get(service, "#1565c0")
    pct     = proba * 100
    bg      = color if is_top else "#90a4ae"
    width   = max(pct, 2)
    txt_in  = f"{pct:.1f}%" if pct > 8 else ""
    bold    = "font-weight:800;" if is_top else ""
    return f"""
    <div class="prob-row">
        <div class="prob-label" style="{bold}">{ICONS.get(service,'')} {service}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{width}%; background:{bg};">
                {txt_in}
            </div>
        </div>
        <div class="prob-value" style="{bold}">{pct:.1f}%</div>
    </div>"""


# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 HospIA")
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Application de **recommandation de services hospitaliers**
    basée sur l'analyse NLP des avis patients.

    **Modèle :** CamemBERT fine-tuned
    **Classes :** 7 services hospitaliers
    **Langue :** Français
    """)
    st.markdown("---")
    st.markdown("### 🏥 Services détectés")
    for svc in SERVICES:
        st.markdown(f"{ICONS[svc]} **{svc}**")
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    show_chart   = st.checkbox("Afficher graphique radar", value=True)
    show_history = st.checkbox("Afficher historique",       value=True)
    threshold    = st.slider("Seuil d'alerte confiance", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown("### 📊 Appareil")
    st.markdown(f"**{'🟢 GPU (CUDA)' if torch.cuda.is_available() else '🔵 CPU'}**")
    st.markdown(f"Device : `{DEVICE}`")
    st.markdown("---")
    st.caption("Projet Master 1 IA — NLP Médical")


# ───────────────────────────────────────────────
# INITIALISATION SESSION STATE
# ───────────────────────────────────────────────
if "historique" not in st.session_state:
    st.session_state.historique = []
if "n_predictions" not in st.session_state:
    st.session_state.n_predictions = 0


# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 HospIA</h1>
    <p>Système de recommandation de services hospitaliers par analyse d'avis patients</p>
    <p style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">
        Propulsé par CamemBERT · NLP · Transformers
    </p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# MÉTRIQUES GLOBALES (top)
# ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-box">
        <div class="val">7</div>
        <div class="lbl">Services détectés</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-box">
        <div class="val">FR</div>
        <div class="lbl">Langue</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-box">
        <div class="val">{st.session_state.n_predictions}</div>
        <div class="lbl">Prédictions</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-box">
        <div class="val">{'GPU' if torch.cuda.is_available() else 'CPU'}</div>
        <div class="lbl">Inférence</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ───────────────────────────────────────────────
with st.spinner("⏳ Chargement du modèle CamemBERT..."):
    try:
        tokenizer, model = load_model()
        st.success("✅ Modèle chargé et prêt pour l'inférence !")
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()


# ───────────────────────────────────────────────
# SECTION PRINCIPALE : SAISIE + RÉSULTATS
# ───────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### ✍️ Saisie de l'avis patient")

    # Exemples prédéfinis
    exemples = {
        "Choisir un exemple...": "",
        "❤️ Cardiologie": "Suite à mon infarctus, j'ai été pris en charge en moins de 10 minutes. Le cardiologue de garde était présent et très compétent. La coronarographie et la pose du stent se sont parfaitement déroulées.",
        "🚨 Urgences": "Arrivée aux urgences pour une douleur thoracique intense à 3h du matin. Triage immédiat, ECG en 5 minutes. Le médecin urgentiste était réactif et rassurant malgré l'affluence importante.",
        "🌸 Gynécologie": "Accouchement dans ce service la semaine dernière. La sage-femme qui m'a accompagnée toute la nuit était extraordinaire, patiente et bienveillante. Je garde un souvenir merveilleux.",
        "👶 Pédiatrie": "Mon fils de 4 ans hospitalisé pour bronchiolite sévère. L'équipe pédiatrique était formidable, douce avec lui et très rassurante pour nous. Suivi impeccable.",
        "🎗️ Oncologie": "Suivi pour cancer du sein depuis 18 mois. L'oncologue est humaine et compétente. La chimiothérapie est bien tolérée grâce au protocole antiémétique adapté.",
        "🦴 Orthopédie": "Prothèse totale de genou posée il y a 6 semaines. L'opération s'est parfaitement déroulée. Le chirurgien était très rassurant avant et après. Rééducation en bonne voie.",
        "🧴 Dermatologie": "Exérèse d'un mélanome suspecté réalisée rapidement. Le dermatologue a été réactif, le geste propre. Suivi anatomopathologique bien organisé.",
    }

    exemple_choisi = st.selectbox(
        "💡 Tester avec un exemple :",
        options=list(exemples.keys())
    )
    texte_exemple = exemples[exemple_choisi]

    avis_patient = st.text_area(
        label="Entrez l'avis du patient en français :",
        value=texte_exemple,
        height=200,
        placeholder="Ex : J'ai été hospitalisé pour des douleurs thoraciques..."
    )

    # Compteur de caractères
    n_chars = len(avis_patient)
    n_mots  = len(avis_patient.split()) if avis_patient.strip() else 0
    col_c1, col_c2 = st.columns(2)
    col_c1.caption(f"📝 {n_chars} caractères")
    col_c2.caption(f"📝 {n_mots} mots")

    if n_chars < 20 and avis_patient.strip():
        st.warning("⚠️ L'avis est très court. Pour de meilleures prédictions, saisissez au moins 20 caractères.")

    # Bouton de prédiction
    predict_btn = st.button("🔍 Analyser l'avis patient", use_container_width=True)

    # Info box
    st.markdown("""
    <div class="info-box">
        💡 <strong>Conseil :</strong> Saisissez un avis détaillé décrivant les symptômes,
        l'expérience vécue ou le type de soins reçus pour une meilleure précision.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────
# RÉSULTATS
# ───────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Résultats de l'analyse")

    if predict_btn and avis_patient.strip():
        texte_propre = nettoyer_texte(avis_patient)

        # Chronomètre
        t_start = time.time()
        with st.spinner("🧠 Analyse en cours..."):
            service, confidence, probas = predict(texte_propre, tokenizer, model)
        t_elapsed = time.time() - t_start

        # Mise à jour compteurs
        st.session_state.n_predictions += 1
        st.session_state.historique.append({
            "avis":      texte_propre[:60] + "...",
            "service":   service,
            "confiance": confidence,
        })

        # ── Résultat principal ──
        icon = ICONS.get(service, "🏥")
        badge = badge_confiance(confidence)
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;">{icon}</div>
            <h2>Service recommandé</h2>
            <div class="service-name">{service}</div>
            <div class="confidence">Confiance : {confidence*100:.1f}%</div>
            {badge}
            <div style="font-size:0.82rem; color:#555; margin-top:0.5rem;">
                ⏱ Inférence en {t_elapsed*1000:.0f} ms
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Description du service
        st.markdown(f"""
        <div class="info-box">
            <strong>{icon} {service} :</strong> {DESCRIPTIONS.get(service, '')}
        </div>
        """, unsafe_allow_html=True)

        # ── Barres de probabilité ──
        st.markdown("#### 📈 Probabilités par service")
        probas_sorted = dict(sorted(probas.items(), key=lambda x: x[1], reverse=True))
        barres_html = ""
        for svc, p in probas_sorted.items():
            barres_html += barre_probabilite(svc, p, svc == service)
        st.markdown(barres_html, unsafe_allow_html=True)

        # ── Alerte faible confiance ──
        if confidence < threshold:
            st.warning(f"⚠️ La confiance ({confidence*100:.1f}%) est sous le seuil défini ({threshold*100:.0f}%). "
                       f"Vérification humaine recommandée.")

    elif predict_btn and not avis_patient.strip():
        st.error("❌ Veuillez saisir un avis patient avant de lancer l'analyse.")

    else:
        # État initial
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#9e9e9e;">
            <div style="font-size:4rem;">🏥</div>
            <p style="font-size:1.1rem;">Saisissez un avis patient et cliquez sur<br>
            <strong>Analyser l'avis patient</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────
# GRAPHIQUE RADAR (optionnel)
# ───────────────────────────────────────────────
if show_chart and predict_btn and avis_patient.strip():
    st.markdown("---")
    st.markdown("### 🕸️ Visualisation Radar des probabilités")

    try:
        probas_vals = [probas[s] for s in SERVICES]
        N = len(SERVICES)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals = probas_vals + probas_vals[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_facecolor('#f8f9ff')
        fig.patch.set_facecolor('#f8f9ff')

        ax.plot(angles, vals, 'o-', linewidth=2.5, color='#1565c0')
        ax.fill(angles, vals, alpha=0.25, color='#1565c0')
        ax.set_xticks(angles[:-1])
        labels = [f"{ICONS[s]}\n{s}" for s in SERVICES]
        ax.set_xticklabels(labels, size=9, color='#1a237e', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["25%","50%","75%","100%"], size=7, color='#888')
        ax.grid(color='#ccc', linestyle='--', linewidth=0.7)
        ax.spines['polar'].set_color('#ddd')
        ax.set_title("Distribution des probabilités", pad=20,
                     fontsize=12, fontweight='bold', color='#1a237e')

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            st.pyplot(fig, use_container_width=True)
        plt.close()

    except Exception as e:
        st.info(f"Graphique radar indisponible : {e}")


# ───────────────────────────────────────────────
# HISTORIQUE DES PRÉDICTIONS
# ───────────────────────────────────────────────
if show_history and st.session_state.historique:
    st.markdown("---")
    st.markdown("### 🕐 Historique des prédictions")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.historique = []
            st.rerun()

    historique_rev = list(reversed(st.session_state.historique))
    for i, item in enumerate(historique_rev[:10]):
        icon    = ICONS.get(item['service'], '🏥')
        conf_pct = item['confiance'] * 100
        badge_c  = "badge-high" if conf_pct >= 75 else ("badge-medium" if conf_pct >= 50 else "badge-low")
        st.markdown(f"""
        <div class="history-item">
            <div style="flex:1; color:#333; font-size:0.88rem;">
                <strong>#{len(st.session_state.historique)-i}</strong>
                &nbsp;·&nbsp; {item['avis']}
            </div>
            <div style="min-width:120px; text-align:right;">
                <span style="font-weight:700; color:#1a237e;">{icon} {item['service']}</span><br>
                <span class="badge {badge_c}">{conf_pct:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(st.session_state.historique) > 10:
        st.caption(f"Affichage des 10 derniers sur {len(st.session_state.historique)} prédictions.")


# ───────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9e9e9e; font-size:0.82rem; padding:1rem 0;">
    🏥 <strong>HospIA</strong> · Projet Master 1 Intelligence Artificielle · NLP Médical<br>
    Modèle : CamemBERT (camembert-base) · Framework : Streamlit + HuggingFace Transformers<br>
    ⚠️ <em>Application à visée pédagogique uniquement — ne remplace pas un diagnostic médical.</em>
</div>
""", unsafe_allow_html=True)