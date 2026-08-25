# -*- coding: utf-8 -*-
"""
prepara_dados.py
Prepara todos os dados de entrada usados nos exercicios ex1a..ex4b.

Para que os exercicios sejam reprodutiveis e auditaveis, toda
aquisicao/sintese de dados fica centralizada aqui:

  1. Par estereo real (aloeL/aloeR, amostras oficiais do OpenCV)  -> ex1a
     (fallback: par estereo sintetico gerado por deslocamento horizontal)
  2. Video sintetico com objeto verde em movimento                -> ex1b
  3. Dataset LFW (Labeled Faces in the Wild, via scikit-learn):
     - fotos de cadastro de 3 identidades                         -> ex2b
     - video "stream" simulado com rostos                         -> ex2a/ex2b
  4. Par de imagens da mesma cena com rotacao+escala+iluminacao   -> ex3a/ex3b

Execucao:  python prepara_dados.py
"""
import os
import urllib.request

import cv2
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


# ---------------------------------------------------------------- 1. estereo
def prepara_estereo():
    """Baixa o par estereo aloeL/aloeR do repositorio oficial do OpenCV.
    E um par retificado classico (Middlebury), ideal para StereoBM/SGBM."""
    base = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
    ok = True
    for nome in ("aloeL.jpg", "aloeR.jpg"):
        destino = os.path.join(DATA, nome)
        if os.path.exists(destino):
            continue
        try:
            print(f"[estereo] baixando {nome} ...")
            urllib.request.urlretrieve(base + nome, destino)
        except Exception as e:
            print(f"[estereo] falha no download ({e}); usando fallback sintetico")
            ok = False
            break
    if ok and cv2.imread(os.path.join(DATA, "aloeL.jpg")) is not None:
        print("[estereo] par aloeL/aloeR pronto")
        return

    # Fallback sintetico: gera par estereo deslocando "camadas" de profundidade.
    from skimage import data as skdata
    img = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    esq = img.copy()
    dir_ = np.zeros_like(img)
    # objetos mais proximos (centro) deslocam mais que o fundo (disparidade maior)
    for x in range(w):
        for faixa, disp in (((0, h // 3), 4), ((h // 3, 2 * h // 3), 12),
                            ((2 * h // 3, h), 24)):
            y0, y1 = faixa
            xs = min(w - 1, x + disp)
            dir_[y0:y1, x] = img[y0:y1, xs]
    cv2.imwrite(os.path.join(DATA, "aloeL.jpg"), esq)
    cv2.imwrite(os.path.join(DATA, "aloeR.jpg"), dir_)
    print("[estereo] par sintetico gerado")


# ------------------------------------------------------- 2. video segmentacao
def prepara_video_bola():
    """Sintetiza um video controlado: bola verde em trajetoria senoidal sobre
    fundo com textura + um distrator vermelho. Video controlado permite validar
    a segmentacao HSV com ground truth conhecido (so a bola e verde)."""
    destino = os.path.join(DATA, "bola.mp4")
    if os.path.exists(destino):
        print("[video] bola.mp4 ja existe")
        return
    w, h, n = 640, 480, 90
    rng = np.random.default_rng(42)
    fundo = rng.integers(90, 140, (h, w, 3), dtype=np.uint8)  # cinza texturizado
    fundo = cv2.GaussianBlur(fundo, (7, 7), 0)
    vw = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if not vw.isOpened():  # fallback de codec no Windows
        destino = os.path.join(DATA, "bola.avi")
        vw = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))
    for i in range(n):
        fr = fundo.copy()
        # distrator vermelho estatico (deve ser rejeitado pelo range HSV)
        cv2.rectangle(fr, (500, 60), (590, 150), (40, 40, 200), -1)
        # bola verde em movimento; raio varia -> proporcao de area varia
        cx = int(80 + (w - 160) * (i / (n - 1)))
        cy = int(h / 2 + 120 * np.sin(2 * np.pi * i / 45))
        r = int(38 + 14 * np.sin(2 * np.pi * i / 30))
        cv2.circle(fr, (cx, cy), r, (60, 200, 70), -1)
        cv2.circle(fr, (cx - r // 3, cy - r // 3), r // 4, (120, 235, 140), -1)
        # ruido sal-e-pimenta leve para exigir morfologia na limpeza da mascara
        ruido = rng.random((h, w))
        fr[ruido > 0.998] = (70, 210, 80)   # pixels verdes espurios
        fr[ruido < 0.002] = (0, 0, 0)
        vw.write(fr)
    vw.release()
    print(f"[video] {os.path.basename(destino)} gerado ({n} frames)")


# ----------------------------------------------------------------- 3. rostos
def prepara_rostos():
    """Baixa LFW via scikit-learn e monta:
       - data/identidades/<Nome>/cadastro_XX.jpg (3 fotos por identidade, ex2b)
       - data/faces_stream.mp4 (video simulando stream de camera, ex2a/ex2b)
    LFW foi escolhido por ser um dataset academico publico com multiplas fotos
    por identidade — nao ha fotos pessoais no enunciado."""
    destino_video = os.path.join(DATA, "faces_stream.mp4")
    dir_ident = os.path.join(DATA, "identidades")
    if os.path.exists(destino_video) and os.path.isdir(dir_ident):
        print("[rostos] dados ja existem")
        return
    from sklearn.datasets import fetch_lfw_people
    print("[rostos] baixando LFW (pode demorar alguns minutos na 1a vez)...")
    lfw = fetch_lfw_people(min_faces_per_person=70, color=True,
                           slice_=None, resize=None,
                           data_home=os.path.join(DATA, "skl"))
    nomes = lfw.target_names
    # 3 identidades com mais fotos
    contagens = np.bincount(lfw.target)
    top3 = np.argsort(contagens)[::-1][:3]
    print("[rostos] identidades:", [nomes[i] for i in top3])

    def para_bgr(img):
        a = np.asarray(img)
        if a.max() <= 1.0:
            a = a * 255.0
        return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)

    os.makedirs(dir_ident, exist_ok=True)
    frames_stream = []
    for ident in top3:
        idxs = np.where(lfw.target == ident)[0]
        pasta = os.path.join(dir_ident, nomes[ident].replace(" ", "_"))
        os.makedirs(pasta, exist_ok=True)
        # 3 primeiras fotos -> cadastro (galeria); proximas 8 -> stream (sonda)
        for k, idx in enumerate(idxs[:3]):
            cv2.imwrite(os.path.join(pasta, f"cadastro_{k:02d}.jpg"),
                        para_bgr(lfw.images[idx]))
        for idx in idxs[3:11]:
            frames_stream.append(para_bgr(lfw.images[idx]))

    # monta o video: cada foto vira 6 frames (simula stream a 10 fps), 2x upscale
    h, w = frames_stream[0].shape[:2]
    vw = cv2.VideoWriter(destino_video, cv2.VideoWriter_fourcc(*"mp4v"),
                         10, (w * 2, h * 2))
    ordem = np.random.default_rng(7).permutation(len(frames_stream))
    for i in ordem:
        fr = cv2.resize(frames_stream[i], (w * 2, h * 2))
        for _ in range(6):
            vw.write(fr)
    vw.release()
    print(f"[rostos] faces_stream.mp4 gerado ({len(frames_stream) * 6} frames)")


# ------------------------------------------------------------- 4. cena (ex3)
def prepara_cena():
    """Gera par de imagens da mesma cena sob condicoes diferentes:
    referencia + versao com rotacao 35 graus, escala 0.75 e brilho reduzido.
    Transformacao conhecida -> permite validar visualmente a homografia."""
    ref = os.path.join(DATA, "cena_ref.png")
    tr = os.path.join(DATA, "cena_transf.png")
    if os.path.exists(ref) and os.path.exists(tr):
        print("[cena] imagens ja existem")
        return
    from skimage import data as skdata
    img = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)  # 512x512, textura rica
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 35, 0.75)
    transf = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    transf = cv2.convertScaleAbs(transf, alpha=0.85, beta=-25)  # iluminacao distinta
    cv2.imwrite(ref, img)
    cv2.imwrite(tr, transf)
    print("[cena] cena_ref.png / cena_transf.png gerados")


if __name__ == "__main__":
    prepara_estereo()
    prepara_video_bola()
    prepara_rostos()
    prepara_cena()
    print("\n[ok] todos os dados prontos em ./data")
