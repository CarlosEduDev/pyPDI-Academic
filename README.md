# 📷 Motor de Processamento Digital de Imagens (PDI) em Python

Este projeto é uma biblioteca/motor de **Processamento Digital de Imagens (PDI)** implementada do zero em Python, utilizando apenas **NumPy** para operações matriciais e **Pillow (PIL)** para E/S (leitura e gravação) de imagens. 

O objetivo do projeto é demonstrar a implementação prática de algoritmos clássicos de PDI em escala de cinza (8 bits/pixel) sem o uso de bibliotecas de alto nível como OpenCV ou Scikit-Image.

---

## 🚀 Funcionalidades Implementadas

O motor conta com um conjunto abrangente de funções divididas por categorias:

### 1. Operações Algébricas
* **Dissolve Cruzado Uniforme:** Mistura de duas imagens via fator ponderado $t \in [0, 1]$.
* **Dissolve Cruzado Não-Uniforme:** Transição baseada em uma matriz/mapa de pesos $T(x,y)$.

### 2. Transformações de Intensidade (Ponto a Ponto)
* **Negativo de Imagem:** Inversão da escala de cinza ($255 - f$).
* **Alargamento de Contraste:** Mapeamento linear para esticar faixas dinâmicas comprimidas.
* **Limiarização (Binarização):** Conversão para imagem binária ($0$ ou $255$) com base em um limiar $T$.
* **Transformação de Potência (Gama):** Correção de brilho não-linear ($s = c \cdot r^\gamma$).
* **Transformação Logarítmica:** Expansão de tons escuros ($s = c \cdot \log(1 + r)$).

### 3. Processamento de Histograma
* **Expansão de Histograma:** Mapeamento automático dos valores $[\min, \max]$ da imagem para $[0, 255]$.
* **Equalização Global de Histograma:** Redistribuição de tons de cinza via Função de Distribuição Acumulada (CDF).

### 4. Processamento Local & Filtragem Espacial
* **Contraste Adaptativo Local:** Amplificação de contraste com base na média de vizinhança $n \times n$.
* **Convolução 2D Genérica:** Motor flexível com suporte a tratamento de bordas (*replicate*, *zero*, *reflect*).
* **Filtro da Média:** Suavização espacial linear para redução de ruído.
* **Filtro da Mediana:** Filtro espacial não-linear ideal para remoção de ruído do tipo "sal e pimenta".

### 5. Detecção de Bordas
* **Operador de Roberts:** Gradiente via máscaras $2 \times 2$ diagonais.
* **Operador de Sobel:** Gradiente via máscaras $3 \times 3$ direcionais ($G_x$ e $G_y$).
* **Operador de Kirsch:** Detecção de bordas considerando 8 direções espaciais.
* **Operador Laplaciano:** Detecção de descontinuidades via segunda derivada (máscara isotrópica).

### 6. Realce e Aguçamento (Sharpening)
* **Aguçamento Laplaciano:** Realce de detalhes somando as bordas Laplacianas à imagem original.
* **High-Boost Filtering:** Realce de altas frequências utilizando máscara de nitidez (*unsharp masking*) ponderada pelo fator $k$.

---

## 🛠️ Pré-requisitos

Para executar este projeto, você precisará de **Python 3.8+** instalado.

### Bibliotecas necessárias:
* `numpy`
* `pillow`

Você pode instalar as dependências via `pip`:

```bash
pip install numpy pillow
