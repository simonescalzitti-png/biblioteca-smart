import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Biblioteca Scolastica", page_icon="📚")

# 1. Caricamento dati (con il trucco per il separatore che abbiamo usato prima)
@st.cache_data
def load_data():
    return pd.read_csv('biblioteca.csv', sep=None, engine='python')

df = load_data()

st.title("📚 Consigli di Lettura su Misura")

# --- NUOVA SEZIONE: FILTRO ETÀ ---
st.sidebar.header("Impostazioni Utente")
eta_utente = st.sidebar.number_input("Quanti anni hai?", min_value=5, max_value=19, value=11)

# Filtriamo subito il database: teniamo solo i libri adatti all'utente
df_filtrato = df[df['eta_minima'] <= eta_utente].reset_index(drop=True)

if df_filtrato.empty:
    st.warning("Ops! Non ci sono ancora libri catalogati per la tua fascia d'età.")
else:
    # 2. Ricalcoliamo la somiglianza solo sui libri filtrati
    tfidf = TfidfVectorizer(stop_words=['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una'])
    tfidf_matrix = tfidf.fit_transform(df_filtrato['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. Selezione libro (tra quelli filtrati)
    libro_scelto = st.selectbox("Quale libro ti è piaciuto tra questi?", df_filtrato['titolo'].values)

    if st.button('Trova il prossimo libro'):
        idx = df_filtrato[df_filtrato['titolo'] == libro_scelto].index[0]
        punteggi = list(enumerate(cosine_sim[idx]))
        punteggi = sorted(punteggi, key=lambda x: x[1], reverse=True)
        
        # Prendiamo i consigli (escludendo se stessi)
        consigli = [p for p in punteggi if p[0] != idx][:3]
        
        st.subheader(f"Dato che hai {eta_utente} anni, ti consigliamo:")
        cols = st.columns(3)
        
        for i, (index, score) in enumerate(consigli):
            with cols[i]:
                st.success(f"**{df_filtrato.iloc[index]['titolo']}**")
                st.caption(f"Età consigliata: {df_filtrato.iloc[index]['eta_minima']}+")
