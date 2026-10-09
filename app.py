import numpy as np
from PIL import Image
import streamlit as st

from pdi_funcoes import (
    alargamento_contraste,
    contraste_adaptivo,
    dissolve_uniforme,
    equalizacao,
    expansao,
    filtro_media,
    filtro_mediana,
    high_boost,
    limiarizacao,
    logaritmo,
    negativo,
    operador_kirsh,
    operador_roberts,
    operador_sobel,
    potencia_gamma,
)


st.set_page_config(
    page_title="PDI Engine - Visualizador Interativo",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Motor de Processamento Digital de Imagens (PDI)")
st.write(
    "Interface interativa para testar algoritmos de PDI implementados em NumPy."
)

# Barra Lateral (Sidebar) - Controles e Seleção
st.sidebar.header("⚙️ Configurações e Parâmetros")

# 1. Upload da Imagem
uploaded_file = st.sidebar.file_uploader(
    "Carregue uma imagem (8-bits / Escala de Cinza)",
    type=["tiff", "png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    # Carrega a imagem enviada pelo usuário
    image_pil = Image.open(uploaded_file).convert("L")
else:
    # Tenta carregar a imagem padrão do projeto se não houver upload
    try:
        image_pil = Image.open("legumes.tiff").convert("L")
        st.sidebar.info("Exibindo imagem padrão: `legumes.tiff`")
    except FileNotFoundError:
        st.warning(
            "Por favor, faça o upload de uma imagem na barra lateral para começar."
        )
        st.stop()

# Converter para Array NumPy 2D
f = np.array(image_pil)

# 2. Seleção do Grupo de Operações
categoria = st.sidebar.selectbox(
    "Escolha a Categoria de Processamento:",
    [
        "Transformações de Intensidade",
        "Histograma",
        "Filtragem Espacial",
        "Detecção de Bordas",
        "Aguçamento (Sharpening)",
        "Contraste Adaptativo",
        "Operações Algébricas (Dissolve)",
    ],
)

# Execução das Operações com base nos parâmetros selecionados
out = f.copy()
subtitulo = "Imagem Original"

if categoria == "Transformações de Intensidade":
    op = st.sidebar.radio(
        "Operação:",
        [
            "Negativo",
            "Limiarização",
            "Alargamento de Contraste",
            "Potência (Gama)",
            "Logaritmo",
        ],
    )

    if op == "Negativo":
        out = negativo(f)
        subtitulo = "Resultado: Negativo"

    elif op == "Limiarização":
        limiar = st.sidebar.slider("Limiar (T):", 0, 255, 128)
        out = limiarizacao(f, limiar)
        subtitulo = f"Resultado: Limiarização (T={limiar})"

    elif op == "Alargamento de Contraste":
        r_min_val = float(np.min(f))
        r_max_val = float(np.max(f))
        r_min = st.sidebar.slider(
            "r_min:", 0.0, 255.0, r_min_val
        )
        r_max = st.sidebar.slider(
            "r_max:", 0.0, 255.0, r_max_val
        )
        out = alargamento_contraste(f, r_min, r_max, 0, 255)
        subtitulo = "Resultado: Alargamento de Contraste"

    elif op == "Potência (Gama)":
        c = st.sidebar.number_input("Constante c:", value=1.0)
        gamma = st.sidebar.slider("Gama (γ):", 0.1, 5.0, 1.0, step=0.1)
        out = potencia_gamma(f, c, gamma)
        subtitulo = f"Resultado: Potência Gama (γ={gamma}, c={c})"

    elif op == "Logaritmo":
        c = st.sidebar.number_input("Constante c:", value=1.0)
        out = logaritmo(f, c)
        subtitulo = f"Resultado: Transformação Logarítmica (c={c})"

elif categoria == "Histograma":
    op = st.sidebar.radio("Operação:", ["Equalização", "Expansão"])
    if op == "Equalização":
        out = equalizacao(f)
        subtitulo = "Resultado: Equalização de Histograma"
    else:
        out = expansao(f)
        subtitulo = "Resultado: Expansão de Histograma"

elif categoria == "Filtragem Espacial":
    op = st.sidebar.radio("Filtro:", ["Média", "Mediana"])
    n = st.sidebar.slider(
        "Tamanho da Vizinhança (n x n):", 3, 15, 3, step=2
    )
    if op == "Média":
        out = filtro_media(f, n)
        subtitulo = f"Resultado: Filtro da Média ({n}x{n})"
    else:
        out = filtro_mediana(f, n)
        subtitulo = f"Resultado: Filtro da Mediana ({n}x{n})"

elif categoria == "Detecção de Bordas":
    op = st.sidebar.radio("Operador:", ["Sobel", "Roberts", "Kirsch"])
    if op == "Sobel":
        gx, gy, mag = operador_sobel(f)
        out = mag
        subtitulo = "Resultado: Magnitude do Operador de Sobel"
    elif op == "Roberts":
        out = operador_roberts(f)
        subtitulo = "Resultado: Magnitude do Operador de Roberts"
    elif op == "Kirsch":
        resps, mag = operador_kirsh(f)
        out = mag
        subtitulo = "Resultado: Magnitude do Operador de Kirsch (8 direções)"

elif categoria == "Aguçamento (Sharpening)":
    c_boost = st.sidebar.slider(
        "Fator de Aguçamento (c):", 0.1, 5.0, 1.0, step=0.1
    )
    n_blur = st.sidebar.slider("Desfoque Base (n x n):", 3, 9, 3, step=2)
    mascara, out = high_boost(f, c=c_boost, n_blur=n_blur)
    subtitulo = f"Resultado: High Boost (c={c_boost}, n={n_blur})"

elif categoria == "Contraste Adaptativo":
    n = st.sidebar.slider(
        "Tamanho da Vizinhança (n x n):", 3, 15, 3, step=2
    )
    c = st.sidebar.slider("Constante c:", 0.1, 5.0, 1.0, step=0.1)
    out = contraste_adaptivo(f, n, c)
    subtitulo = f"Resultado: Contraste Adaptativo (n={n}, c={c})"

elif categoria == "Operações Algébricas (Dissolve)":
    t = st.sidebar.slider("Fator de Mistura (t):", 0.0, 1.0, 0.5, step=0.05)
    # Gera uma imagem secundária invertida para testar o dissolve
    g = 255 - f
    out = dissolve_uniforme(f, g, t)
    subtitulo = f"Resultado: Dissolve Uniforme com Negativo (t={t})"

# Exibição Visual Lado a Lado no Streamlit
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original (8-bits)")
    st.image(
        f,
        caption=f"Resolução: {f.shape[1]}x{f.shape[0]} pixels",
        use_column_width=True,
    )

with col2:
    st.subheader(subtitulo)
    st.image(out, use_column_width=True)