import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurazione pagina
st.set_page_config(page_title="Biblioteca Scolastica", page_icon="📚")

st.title("📚 Consigli di Lettura Smart")
st.markdown("Scegli un libro che ti è piaciuto e ti dirò cosa leggere dopo!")

# Caricamento dati
@st.cache_data # Questo serve per rendere l'app velocissima
def load_data():
    return pd.read_csv('biblioteca.csv')

df = load_data()

# Calcolo Algoritmo
tfidf = TfidfVectorizer(stop_words=['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una'])
tfidf_matrix = tfidf.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Interfaccia
libro_scelto = st.selectbox("Cerca un libro:", df['titolo'].values)

if st.button('Ottieni Consigli'):
    idx = df[df['titolo'] == libro_scelto].index[0]
    punteggi = list(enumerate(cosine_sim[idx]))
    punteggi = sorted(punteggi, key=lambda x: x[1], reverse=True)[1:4] # Top 3 consigli
    
    st.subheader("Ti potrebbero piacere questi:")
    cols = st.columns(3) # Crea tre colonne per i risultati
    
    for i, (index, score) in enumerate(punteggi):
        with cols[i]:
            st.success(f"**{df.iloc[index]['titolo']}**")
            st.info(f"Autore: {df.iloc[index]['autore']}")