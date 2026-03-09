# 🏥 Recommandation de Services Hospitaliers par NLP

> **Projet académique** — Analyse d'avis patients et prédiction automatique du service hospitalier approprié grâce au Traitement Automatique du Langage Naturel (NLP).

---

## 📋 Présentation du projet

Ce projet démontre comment des techniques de NLP peuvent être appliquées à des avis médicaux de patients pour recommander automatiquement un service hospitalier (cardiologie, psychiatrie, dermatologie, etc.).

Le pipeline complet couvre :

- L'exploration et le nettoyage d'un dataset réel de reviews médicales
- Le préprocessing NLP (tokenisation, stopwords, lemmatisation)
- La vectorisation TF-IDF
- L'entraînement de 3 modèles de complexité croissante
- Une application web interactive déployée sur Streamlit

---

## 📂 Structure du projet

```
project/
├── dataset/
│   └── drugsComTest_raw.csv          # Dataset UCI/Kaggle
├── notebook/
│   └── nlp_hospital_recommendation.ipynb  # Notebook Colab complet
├── models/
│   ├── model_ml.pkl                  # Logistic Regression (app principale)
│   ├── tfidf_vectorizer.pkl          # Vectoriseur TF-IDF
│   ├── label_encoder.pkl             # Encodeur des labels
│   ├── model_dl.h5                   # Neural Network (Keras)
│   ├── keras_tokenizer.json          # Tokenizer Keras
│   └── bert_model_dir/               # DistilBERT fine-tuné
├── app.py                            # Application Streamlit
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

---

## 📊 Dataset

| Propriété              | Valeur                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| **Source**             | [Drug Review Dataset — Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) |
| **Origine**            | UCI Machine Learning Repository                                                                         |
| **Taille**             | ~53 000 avis patients                                                                                   |
| **Colonnes utilisées** | `review` (texte), `condition` (maladie → service)                                                       |

### Mapping Condition → Service hospitalier

| Service                 | Exemples de conditions                                |
| ----------------------- | ----------------------------------------------------- |
| 🧘 **Psychiatry**       | Depression, Anxiety, ADHD, Insomnia, Bipolar Disorder |
| ❤️ **Cardiology**       | High Blood Pressure, Heart Failure, High Cholesterol  |
| 🌸 **Gynecology**       | Birth Control, Emergency Contraception, Endometriosis |
| 🩹 **Dermatology**      | Acne, Eczema, Psoriasis, Rosacea                      |
| ⚖️ **Endocrinology**    | Obesity, Diabetes Type 2, Hypothyroidism, Weight Loss |
| 🧠 **Neurology**        | Migraine, Epilepsy, Multiple Sclerosis                |
| 🫁 **Pulmonology**      | Asthma, COPD, Smoking Cessation                       |
| 🦴 **Rheumatology**     | Pain, Fibromyalgia, Rheumatoid Arthritis, Back Pain   |
| 🫀 **Gastroenterology** | Crohn's Disease, IBS, GERD, Ulcerative Colitis        |
| 💧 **Urology**          | Urinary Tract Infection, Vaginal Yeast Infection      |

---

## 🔬 Méthodologie

### Pipeline NLP

```
Texte brut
    ↓
Décodage HTML entities
    ↓
Mise en minuscules
    ↓
Suppression ponctuation & chiffres
    ↓
Suppression stopwords (NLTK)
    ↓
Lemmatisation (WordNetLemmatizer)
    ↓
Vectorisation TF-IDF (20 000 features, unigrams + bigrams)
    ↓
Classification
```

### Les 3 modèles

#### 1️⃣ Logistic Regression (Baseline)

- Vectorisation : TF-IDF
- Simple, rapide, interprétable
- **Utilisé dans l'application Streamlit**

#### 2️⃣ Neural Network (Keras)

- Architecture : `Embedding → GlobalAveragePooling → Dense(128) → Dropout → Softmax`
- Léger, entraînable sur CPU en < 5 minutes
- Apprend des représentations vectorielles denses

#### 3️⃣ DistilBERT (HuggingFace)

- Transformer pré-entraîné, fine-tuné sur le dataset
- `max_length=128`, 3 epochs
- Sous-échantillonnage à 5 000 exemples pour Colab gratuit

---

## 📈 Résultats (indicatifs)

| Modèle                 | Accuracy | F1-score (weighted) | Temps entraînement |
| ---------------------- | -------- | ------------------- | ------------------ |
| Logistic Regression    | ~0.87    | ~0.86               | < 30s              |
| Neural Network (Keras) | ~0.83    | ~0.82               | ~3 min             |
| DistilBERT             | ~0.90    | ~0.89               | ~15 min (GPU)      |

> ⚠️ Les performances varient selon le split train/test. DistilBERT est évalué sur un sous-ensemble.

---

## 🚀 Lancer l'application

### Prérequis

```bash
pip install -r requirements.txt
```

### 1. Entraîner les modèles (Google Colab)

1. Ouvrir `notebook/nlp_hospital_recommendation.ipynb` dans Google Colab
2. Activer le GPU : `Runtime → Change runtime type → T4 GPU`
3. Uploader le dataset ou le monter depuis Google Drive
4. Exécuter toutes les cellules
5. Télécharger les fichiers générés : `model_ml.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`
6. Les placer dans le dossier `models/`

### 2. Lancer l'app localement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

### 3. Déployer sur Streamlit Cloud

1. Pousser le projet sur GitHub (inclure le dossier `models/`)
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le dépôt GitHub
4. Sélectionner `app.py` comme fichier principal
5. Cliquer sur **Deploy**

---

## 💡 Exemple d'utilisation

**Entrée :**

```
I have strong chest pains and shortness of breath, my heart is racing.
```

**Sortie :**

```
❤️ Service recommandé : Cardiology
Confiance : 92%
```

---

## ⚠️ Avertissement

> Ce projet est réalisé à des fins **académiques uniquement**.  
> Les recommandations générées ne constituent pas un avis médical.  
> En cas de problème de santé, consultez un professionnel de santé qualifié.

---

## 🛠️ Technologies utilisées

| Outil                    | Usage                        |
| ------------------------ | ---------------------------- |
| Python 3.10+             | Langage principal            |
| scikit-learn             | TF-IDF, Logistic Regression  |
| TensorFlow/Keras         | Neural Network               |
| HuggingFace Transformers | DistilBERT                   |
| NLTK                     | Préprocessing NLP            |
| Streamlit                | Application web              |
| Google Colab             | Environnement d'entraînement |

---

## 📚 Références

- [Drug Review Dataset — UCI/Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
- [DistilBERT — HuggingFace](https://huggingface.co/distilbert-base-uncased)
- [Streamlit Documentation](https://docs.streamlit.io)
- Gräßer et al. (2018). _Aspect-Based Sentiment Analysis of Drug Reviews_

---

_Projet NLP Académique · Recommandation de Services Hospitaliers_
