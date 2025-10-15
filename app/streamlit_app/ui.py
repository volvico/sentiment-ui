import os
import re
import time as pytime
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Import local (évite soucis d'import package avec Streamlit)
from client import get_health, predict_one, explain_lime

st.set_page_config(page_title="Analyse de Sentiments", page_icon="💬", layout="centered")

# ------------------ STYLES (compteur & tags) ------------------
st.markdown("""
<style>
.char-counter {
  font-size: 0.85rem; margin-top: 0.25rem;
}
.char-green { color: #1aa260; }   /* < 240 */
.char-yellow { color: #f6c343; }  /* 240-280 */
.char-red { color: #e85d5d; }     /* > 280 */
.badge { display:inline-block; padding: 0.2rem 0.5rem; border-radius: 0.5rem; background:#eef2f7; margin-right:0.25rem;}
.small { font-size: 0.9rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Base URL API")
    st.code(os.environ.get("API_BASE_URL", "http://localhost:8000"), language="text")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🔎 Vérifier API"):
            try:
                h = get_health()
                st.success(f"OK — {h.get('model_class')} (mode: {h.get('mode')})")
                st.caption(f"Vectorizer: {h.get('vectorizer_class')}")
            except Exception as e:
                st.error(f"API DOWN: {e}")
    with colB:
        st.write("")

    st.markdown("---")
    st.subheader("🧪 Exemples")
    examples = [
        "J'adore ce produit, il est fantastique !",
        "Service déplorable, je suis très déçu.",
        "Ce film est excellent, un vrai coup de cœur ❤️",
        "C'est une perte de temps. Horrible expérience.",
    ]
    chosen = st.selectbox("Insérer un exemple :", ["(aucun)"] + examples)
    if chosen != "(aucun)":
        st.session_state["prefill_text"] = chosen

    st.markdown("---")
    st.subheader("📝 Guide d’utilisation")
    st.markdown("""
- Saisir un texte (max 280 caractères).
- **Prédire** pour obtenir le sentiment & probabilités.
- **LIME** pour l'explication visuelle (peut prendre 30–60 s).
""")

# ------------------ MAIN HEADER ------------------
st.title("💬 Interface d’Analyse de Sentiments")
st.caption("API FastAPI + Streamlit — explications LIME et probabilités")

# ------------------ ZONE DE SAISIE ------------------
default_text = st.session_state.get("prefill_text", "")
tweet_text = st.text_area(
    "Votre texte (≤ 280 caractères)",
    value=default_text,
    max_chars=280,
    height=140,
    placeholder="Tapez votre avis ici...",
    help="Les emojis/URLs sont acceptés. La prédiction est binaire (Positif/Négatif)."
)

# Compteur couleur
n = len(tweet_text)
if n <= 240:
    cls = "char-green"
elif n <= 280:
    cls = "char-yellow"
else:
    cls = "char-red"
st.markdown(f'<div class="char-counter {cls}">{n} / 280</div>', unsafe_allow_html=True)

# Validation côté client
text_has_word_char = bool(re.search(r"\w", tweet_text))
text_valid = (len(tweet_text.strip()) > 0) and (len(tweet_text) <= 280) and text_has_word_char

# ------------------ BOUTONS D'ACTION ------------------
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🎯 Prédire Sentiment", type="primary", disabled=not text_valid)
with col2:
    explain_btn = st.button("🔍 LIME (30–60s)", disabled=not text_valid)

# Espace résultats
st.markdown("---")
res_col1, res_col2 = st.columns([1,1])

# ------------------ PREDICTION ------------------
if predict_btn:
    with st.spinner("Prédiction en cours..."):
        t0 = pytime.time()
        try:
            resp = predict_one(tweet_text)
            if resp["status_code"] == 200 and resp["json"] is not None:
                data = resp["json"]
                sentiment = data["sentiment"]
                conf = float(data["confidence"])
                p_pos = float(data["probability_positive"])
                p_neg = float(data["probability_negative"])

                with res_col1:
                    if sentiment.lower().startswith("pos"):
                        st.success(f"😊 **POSITIF** ({conf:.1%})")
                    else:
                        st.error(f"😞 **NÉGATIF** ({conf:.1%})")
                    st.markdown(f'<span class="badge">p(Positif): {p_pos:.2f}</span> <span class="badge">p(Négatif): {p_neg:.2f}</span>', unsafe_allow_html=True)
                    st.caption(f"Durée: {(time.time()-t0):.2f}s")

                with res_col2:
                    fig = px.bar(
                        x=['Négatif', 'Positif'],
                        y=[p_neg, p_pos],
                        labels={'x':'Classe', 'y':'Probabilité'},
                        title='Probabilités'
                    )
                    fig.update_yaxes(range=[0,1])
                    st.plotly_chart(fig, use_container_width=True)

            elif resp["status_code"] == 422:
                st.warning("Entrée invalide (422). Vérifie la taille (< 280) et le contenu.")
            else:
                st.error(f"Erreur API (predict): {resp['status_code']} — {resp['text']}")
        except Exception as e:
            st.error(f"Erreur de communication: {e}")

# ------------------ LIME ------------------
if explain_btn:
    # Petit “progress bar” pour indiquer que ça peut durer
    progress = st.progress(0, text="LIME en cours…")
    try:
        for i in range(5):
            pytime.sleep(0.2)
            progress.progress(min((i+1)*20, 100), text="LIME en cours…")

        with st.spinner("Génération de l'explication LIME…"):
            r = explain_lime(tweet_text)
        progress.empty()

        if r["status_code"] == 200 and r["json"] is not None:
            data = r["json"]
            # bandeau résultat principal
            st.subheader("🧠 Explicabilité (LIME)")
            st.caption("Visualisation interactive LIME (HTML)")
            components.html(data["html_explanation"], height=420, scrolling=True)

            with st.expander("📊 Détails"):
                st.write("**Sentiment**:", data.get("sentiment"))
                st.write("**Confiance**:", f"{float(data.get('confidence', 0.0)):.1%}")
                st.write("**Features LIME**:", data.get("explanation", []))

        elif r["status_code"] == 501:
            st.warning("LIME non installé côté API (501). Ajoute `lime` dans requirements côté serveur.")
        elif r["status_code"] == 422:
            st.warning("Entrée invalide (422). Vérifie la taille (< 280) et le contenu.")
        else:
            st.error(f"Erreur API (LIME): {r['status_code']} — {r['text']}")
    except Exception as e:
        progress.empty()
        st.error(f"Erreur LIME: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("© TP MLOps — API FastAPI + Tests Pytest + UI Streamlit")
