import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Biblioteca Smart", page_icon="📚", layout="wide")

@st.cache_data
def load_data():
    # Il trucco sep=None serve per gestire sia virgole che punti e virgola di Excel
    return pd.read_csv('biblioteca.csv', sep=None, engine='python')

df = load_data()

st.title("📚 Sistema di Raccomandazione Bibliotecaria")
st.markdown("---")

# --- SIDEBAR PER FILTRI ---
st.sidebar.header("🔍 Filtra per Studente")
eta_utente = st.sidebar.slider("Quanti anni hai?", 5, 18, 12)

# Estraiamo i mood dal CSV
mood_disponibili = df['mood'].unique().tolist()
mood_scelto = st.sidebar.selectbox("Come ti senti oggi?", ["Qualsiasi"] + mood_disponibili)

# --- LOGICA DI FILTRAGGIO ---
# 1. Filtro Età
df_filtrato = df[df['eta_minima'] <= eta_utente]

# 2. Filtro Mood
if mood_scelto != "Qualsiasi":
    df_filtrato = df_filtrato[df_filtrato['mood'] == mood_scelto]

df_filtrato = df_filtrato.reset_index(drop=True)

# --- INTERFACCIA PRINCIPALE ---
if df_filtrato.empty:
    st.warning("Non abbiamo trovato libri per questi filtri. Prova ad aumentare l'età o cambiare mood!")
else:
    # Preparazione Motore (TF-IDF sui Tags)
    tfidf = TfidfVectorizer(stop_words=['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una'])
    tfidf_matrix = tfidf.fit_transform(df_filtrato['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Scelta del libro base
    libro_scelto = st.selectbox("Quale libro hai letto e ti è piaciuto?", df_filtrato['titolo'].values)

    if st.button('Genera Consigli'):
        idx = df_filtrato[df_filtrato['titolo'] == libro_scelto].index[0]
        
        # Calcolo somiglianza
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Prendiamo i primi 3 libri diversi da quello scelto
        libri_consigliati = [i for i in sim_scores if i[0] != idx][:3]

        st.markdown("### ✨ Ecco cosa dovresti leggere dopo:")
        
        cols = st.columns(3)
        for i, (index, score) in enumerate(libri_consigliati):
            libro = df_filtrato.iloc[index]
            with cols[i]:
                st.info(f"#### {libro['titolo']}")
                st.write(f"**Autore:** {libro['autore']}")
                st.write(f"**Mood:** {libro['mood']}")
                
                # Visualizzazione Recensione
                st.markdown(f"💬 *'{libro['recensione']}'*")
                
                # Feedback simulato (Stelline)
                st.write("⭐" * 5)
