import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurazione della pagina
st.set_page_config(page_title="Biblioteca Smart", page_icon="📚", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('biblioteca.csv', sep=None, engine='python')

df = load_data()

st.title("📚 Il Tuo Consulente Letterario")
st.markdown("Trova il libro perfetto in base alla tua età e al tuo umore.")
st.markdown("---")

# --- SIDEBAR PER FILTRI ---
st.sidebar.header("🔍 Personalizza")
eta_utente = st.sidebar.slider("Quanti anni hai?", 5, 18, 11)

mood_disponibili = df['mood'].unique().tolist()
mood_scelto = st.sidebar.selectbox("Come ti senti oggi?", ["Qualsiasi"] + mood_disponibili)

# --- LOGICA DI FILTRAGGIO ---
df_filtrato = df[df['eta_minima'] <= eta_utente]

if mood_scelto != "Qualsiasi":
    df_filtrato = df_filtrato[df_filtrato['mood'] == mood_scelto]

df_filtrato = df_filtrato.reset_index(drop=True)

# --- VISUALIZZAZIONE RISULTATI ---
if df_filtrato.empty:
    st.warning("Nessun libro trovato. Prova a cambiare i filtri nella colonna a sinistra!")
else:
    # Motore di calcolo
    tfidf = TfidfVectorizer(stop_words=['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una'])
    tfidf_matrix = tfidf.fit_transform(df_filtrato['tags'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    libro_scelto = st.selectbox("Scegli un libro che ti è piaciuto:", df_filtrato['titolo'].values)

    if st.button('Genera Consigli ✨'):
        idx = df_filtrato[df_filtrato['titolo'] == libro_scelto].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        libri_consigliati = [i for i in sim_scores if i[0] != idx][:3]

        if not libri_consigliati:
            st.info("Abbiamo trovato solo questo libro per questa categoria!")
        else:
            st.markdown("### 🎯 Ecco i libri scelti per te:")
            cols = st.columns(3)
            for i, (index, score) in enumerate(libri_consigliati):
                libro = df_filtrato.iloc[index]
                with cols[i]:
                    st.success(f"**{libro['titolo']}**")
                    st.write(f"✍️ **Autore:** {libro['autore']}")
                    
                    # Mostriamo solo la quantità
                    qta = int(libro['quantita'])
                    if qta > 0:
                        st.write(f"📦 **Disponibilità:** {qta} copie")
                    else:
                        st.error("❌ Al momento in prestito")
                    
                    st.caption(f"📈 Affinità: {round(score * 100)}%")
