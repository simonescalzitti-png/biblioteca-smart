import streamlit as st
import pandas as pd
import random

# Configurazione minima
st.set_page_config(page_title="Catalogo Libri", page_icon="📖")

@st.cache_data
def load_data():
    # Carica il CSV rispettando l'ordine delle tue colonne
    return pd.read_csv('biblioteca.csv', sep=None, engine='python')

df = load_data()

st.title("📖 Catalogo della Biblioteca")
st.markdown("Cerca un libro e controlla se è disponibile.")

# --- INTERFACCIA ---
if df.empty:
    st.warning("Il catalogo è vuoto.")
else:
    # Selezione del libro
    libro_scelto = st.selectbox("Seleziona un libro per vedere i dettagli:", df['titolo'].values)
    
    # Recupero info del libro selezionato
    info_libro = df[df['titolo'] == libro_scelto].iloc[0]
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📗 {info_libro['titolo']}")
        st.write(f"✍️ **Autore:** {info_libro['autore']}")
        
        # --- ORDINE MODIFICATO ---
        st.write(f"👶 **Età consigliata:** {info_libro['eta_minima']}+ anni")
        st.write(f"🎭 **Genere:** {info_libro['genere']}")
    
    with col2:
        qta = int(info_libro['quantita'])
        if qta > 0:
            st.success(f"✅ Disponibile ({qta} copie)")
        else:
            st.error("❌ Al momento in prestito")

    st.markdown("---")
    
    # Funzione "Altri libri di questo autore"
    autore_scelto = info_libro['autore']
    altri_libri = df[(df['autore'] == autore_scelto) & (df['titolo'] != libro_scelto)]
    
    if not altri_libri.empty:
        st.write(f"📚 **Altri libri di {autore_scelto}:**")
        for t in altri_libri['titolo']:
            st.write(f"- {t}")
    else:
        # Suggerimenti casuali se non ci sono altri libri dello stesso autore
        st.write("🎲 **Ti suggeriamo anche:**")
        suggerimenti_casuali = df[df['titolo'] != libro_scelto].sample(min(len(df)-1, 3))
        for t in suggerimenti_casuali['titolo']:
            st.write(f"- {t}")
