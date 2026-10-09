import numpy as np
from PIL import Image

def clip_image(img_float):
    return np.clip(img_float, 0, 255).astype(np.uint8)

# Operações algebricas
    # Dissolve uniforme
def dissolve_uniforme(f, g, t): # (1 - t)*f + t*g
    f_d = f.astype(np.float64)
    g_d = g.astype(np.float64)
    out = (1.0 - t) * f_d + t * g_d
    return clip_image(out);

    # Dissolve nao uniforme
def dissolve_nao_uniforme(f, g, mapa_t): # mapa_t é matriz de pesos entre 0 e 1
    f_d = f.astype(np.float64)
    g_d = g.astype(np.float64)
    mapa_d = mapa_t.astype(np.float64)
    out = (1.0 - mapa_d) * f_d + mapa_d * g_d
    return clip_image(out);

# Transformações de intensidade

def negativo(f): # inverte escala de cinza
    return 255 - f

# Estica a faixa [r_min, r_max] pra [s_min, s_max]
def alargamento_contraste(f, r_min, r_max, s_min = 0, s_max = 255):
    f_d = f.astype(np.float64)
    out = (f_d - r_min) * ((r_max - r_min) / (s_max - s_min)) + s_min
    return clip_image(out);

def limiarizacao(f, limiar): # Binariza a imagem, pixels com valor >= limiar recebem 255, caso contrario recebem 0
    out = np.where(f >= limiar, 255, 0)
    return out.astype(np.uint8)

def potencia_gamma(f, c, gamma): #Transformação de potência: s = c * r^gamma
    f_norm = f.astype(np.float64) / 255
    out = c * (f_norm ** gamma) * 255
    return out.astype(np.uint8)

def logaritmo(f, c = 1.0):
    f_norm = f.astype(np.float64) / 255
    out = c * np.log(1.0 + f_norm) * (255 / np.log(2.0))
    return clip_image(out)

# Histograma
def expansao(f):
    r_min = float(np.min(f))
    r_max = float(np.max(f))

    if(r_max == r_min):
        return f.copy()
    return alargamento_contraste(f, r_min, r_max, 0, 255);

def equalizacao(f):
    altura, largura = f.shape # Altura e Largura da imagem
    n_pixels = altura * largura

    hist = np.zeros(256, dtype= int)
    for val in f.flat:
        hist[val] += 1

    cdf = np.cumsum(hist)
    t_map = np.round(255 * cdf / n_pixels).astype(np.uint8)
    
    out = t_map[f]

    return out

# Contraste adaptativo
def contraste_adaptivo(f, n, c):
    pad = n // 2
    f_d = f.astype(np.float64)
    f_padded = np.pad(f_d, pad, mode = 'edge')
    alt, larg = f.shape

    out = np.zeros_like(f_d)

    for i in range(alt):
        for j in range(larg):
            regiao = f_padded[i : i + n, j + n]
            m_local = np.mean(regiao)

            out[i, j] = c * (f_d[i, j] - m_local) + m_local

    return clip_image(out)

# Convolução genérica
def convolucao(f, h_kernel, offset=0, tratamento_borda='replicate'): # Convolução bidimensional genérica: g = f * h + offset.
    kh, kw = h_kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    f_d = f.astype(np.float64)

    if tratamento_borda == 'zero':
        f_padded = np.pad(f_d, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif tratamento_borda == 'reflect':
        f_padded = np.pad(f_d, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:
        f_padded = np.pad(f_d, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    kernel_inv = np.flip(h_kernel)
    out_h, out_w = f.shape
    g = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            regiao = f_padded[i : i + kh, j : j + kw]
            g[i, j] = np.sum(regiao * kernel_inv) + offset

    return g

def filtro_media(f, n):
    kernel = np.ones((n, n) ,dtype = np.float64) / (n*n)

    g = convolucao(f, kernel, offset=0, tratamento_borda='replicate')

    return clip_image(g)

def filtro_mediana(f, n): # filtro de mediana com vizinhança n x n
    pad = n // 2

    f_padded = np.pad(f, pad, mode = 'edge')
    alt, larg = f.shape

    saida = np.zeros((alt, larg), dtype = np.uint8)

    for i in range(alt):
        for j in range(larg):
            regiao = f_padded[i : i + n, j : j + n]
            saida[i, j] = np.median(regiao)

    return saida

def operador_roberts(f):
    rx = np.array([[-1, 0], [0, 1]], dtype = np.float64)
    ry = np.array([[0, -1], [1, 0]], dtype = np.float64)

    gx = convolucao(f, rx)
    gy = convolucao(f, ry)

    g = np.sqrt(gx**2 + gy**2)

    mag = np.hypot(gx, gy)

    return clip_image(mag)

def operador_sobel(f):
    sx = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]], dtype = np.float64)
    sy = np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float64)

    gx = convolucao(f, sx)
    gy = convolucao(f, sy)

    mag = np.hypot(gx, gy)
    
    return gx, gy, clip_image(mag)

def operador_kirsh(f):
    k = [
        np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float64),
        np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float64),
        np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float64),
        np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float64),
        np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float64),
        np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float64),
        np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float64),
        np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float64),
    ]

    respostas = [convolucao(f, mask) for mask in k]
    mag = np.maximum.reduce([np.abs(resp) for resp in respostas])

    return respostas, clip_image(mag)

def operador_laplaciano(f):
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1,0 ]], dtype = np.float64)

    return convolucao(f, lap_kernel)


def agucamento_laplaciano(f, c = -1.0):
    lap = operador_laplaciano(f)

    f_d = f.astype(np.float64)
    g_sharp = f_d + c * lap

    return clip_image(np.abs(lap)), clip_image(g_sharp)

def high_boost(f, c = 1.0, n_blur = 3):
    f_d = f.astype(np.float64)
    kernel_blur = np.ones((n_blur, n_blur), dtype = np.float64) / (n_blur * n_blur)

    f_suavizada = convolucao(f, kernel_blur)
    mascara = f_d - f_suavizada

    g_hb = f_d + c * mascara # g_hb = high booost
    return clip_image(mascara), clip_image(g_hb)

caminho_imagem = 'lena_color.tif'

imagem_pil = Image.open(caminho_imagem).convert('L') # converte para escala de cinza

# Vai converter a imagem do formato PIL para uma array Numpy
f = np.array(imagem_pil)

print("imagem carregada, resolução: " , f.shape[0], 'x' , f.shape[1], 'pixels')

f.shape

img_equalizada = equalizacao(f)

img_mediana = filtro_mediana(f , 3)

gx, gy, magnitude_sobel = operador_sobel(f)

img_binaria = limiarizacao(f, 128)
img_contraste = contraste_adaptivo(f, 50, 200)

mascara_lap, img_agucada_lap = agucamento_laplaciano(f, c=-1.0)

mascara_hb, img_agucada = high_boost(f, c = 1.0, n_blur = 3)

# Image.fromarray(magnitude_sobel).save('magnitude_sobel.png')
# Image.fromarray(img_equalizada).save('Imagem equalizada.png')
# Image.fromarray(img_mediana).save('Imagem mediana.png')
# Image.fromarray(img_binaria).save('Imagem binaria.png')
# Image.fromarray(img_agucada).save("Imagem agucada.png")
Image.fromarray(img_contraste).save("Image contraste.png")
Image.fromarray(mascara_hb).save("Image mascara_hb.png")
Image.fromarray(img_agucada_lap).save("Image agucada_lap.png")