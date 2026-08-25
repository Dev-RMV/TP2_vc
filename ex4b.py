# -*- coding: utf-8 -*-
"""
ex4b.py — CNN como extratora de features + PCA 2D + comparacao com ORB.

(1) Carrega a CNN treinada no ex4a e remove a camada de classificacao,
    extraindo o vetor de 128-d da penultima camada Dense ("features");
(2) Extrai features de 20 imagens de teste (4 por classe, 5 classes);
(3) PCA para 2D e scatter plot colorido por classe;
(4) Compara com descritores ORB extraidos das MESMAS imagens (mesmo metodo
    do exercicio 3).

Execucao:  python ex4b.py   (requer ex4a executado antes)
O scatter plot e exibido em janela ao final e tambem salvo em ./outputs.
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "outputs")

CLASSES = ["camiseta", "calca", "pulover", "vestido", "casaco",
           "sandalia", "camisa", "tenis", "bolsa", "bota"]
# 5 classes visualmente distintas x 4 imagens = 20 imagens (>=4 por classe)
CLASSES_USADAS = [0, 1, 5, 7, 8]      # camiseta, calca, sandalia, tenis, bolsa
POR_CLASSE = 4


def main():
    caminho = os.path.join(OUT, "modelo_fmnist.keras")
    assert os.path.exists(caminho), "Rode antes: python ex4a.py"
    modelo = keras.models.load_model(caminho)

    # (1) extratora: mesma entrada, saida na penultima Dense (128-d).
    # Remover a softmax importa porque a ultima camada colapsa a
    # representacao nas 10 probabilidades de classe; a penultima preserva
    # o espaco de features generico, reutilizavel para outras tarefas.
    # No Keras 3, um Sequential recarregado nao expoe .input/.output;
    # a extratora e reconstruida encadeando simbolicamente as camadas
    # (com os pesos treinados) ate a camada "features", exclusive softmax:
    entrada = keras.Input((28, 28, 1))
    x = entrada
    for camada in modelo.layers:
        x = camada(x)
        if camada.name == "features":
            break
    extratora = keras.Model(entrada, x)

    (_, _), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
    idxs, rotulos = [], []
    for c in CLASSES_USADAS:
        idxs.extend(np.where(y_te == c)[0][:POR_CLASSE])
        rotulos.extend([c] * POR_CLASSE)
    imgs = x_te[idxs]                      # 20 imagens uint8 28x28
    rotulos = np.array(rotulos)

    # (2) features CNN
    feats = extratora.predict((imgs / 255.0).astype("float32")[..., None],
                              verbose=0)
    print("=== ex4b: features CNN + PCA ===")
    print(f"features extraidas: {feats.shape}  (20 imagens x 128-d)")

    # (3) PCA 2D + scatter
    pca = PCA(n_components=2)
    p2 = pca.fit_transform(feats)
    print(f"variancia explicada pelos 2 PCs: "
          f"{pca.explained_variance_ratio_.sum():.1%}")
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in CLASSES_USADAS:
        m = rotulos == c
        ax.scatter(p2[m, 0], p2[m, 1], s=80, label=CLASSES[c])
    ax.set(title="ex4b — features da CNN (128-d) projetadas em 2D via PCA",
           xlabel="PC1", ylabel="PC2")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ex4b_pca_scatter.png"), dpi=120)
    print("scatter salvo em outputs/ex4b_pca_scatter.png")

    # (4) ORB nas MESMAS 20 imagens (28x28 ampliadas 4x, pois o ORB precisa
    # de area minima para a piramide de escalas e o patch BRIEF de 31px)
    orb = cv2.ORB_create(nfeatures=100)
    kps_por_img = []
    for img in imgs:
        grande = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        kp, des = orb.detectAndCompute(grande, None)
        kps_por_img.append(0 if des is None else len(kp))
    print(f"ORB nas mesmas imagens: keypoints por imagem = {kps_por_img}")
    print(f"  media {np.mean(kps_por_img):.1f} kp/imagem; "
          f"{sum(1 for k in kps_por_img if k == 0)} imagens sem nenhum kp")

    # ------------------------------------------------------------------
    # SEPARABILIDADE E COMPARACAO COM ORB (comentarios exigidos):
    #
    # Separabilidade: no scatter PCA, classes de silhueta distinta
    # (calca, bolsa, tenis/sandalia) formam agrupamentos claros e
    # afastados; sobreposicao residual ocorre entre categorias
    # semanticamente proximas (ex. tenis x sandalia). Boa separacao em
    # apenas 2 PCs (de 128 dims) indica que a CNN aprendeu representacoes
    # DISCRIMINATIVAS — a distancia no espaco de features codifica a
    # semantica da classe, o que explica por que features de CNN sao
    # reutilizaveis (transfer learning, retrieval, clustering).
    #
    # Comparacao com ORB (exercicio 3): o ORB descreve VIZINHANCAS LOCAIS
    # de pontos de alto gradiente — em imagens 28x28 de baixa textura ele
    # encontra pouquissimos (ou nenhum) keypoint, como impresso acima, e
    # seus descritores binarios de 32 B descrevem cantos, nao a categoria
    # do objeto: dois tenis diferentes nao geram descritores proximos.
    # A CNN, ao contrario, produz UM vetor global de 128-d treinado para
    # ser semanticamente separavel. Em robotica: ORB serve para geometria
    # (matching, SLAM, pose — exercicio 3); features de CNN servem para
    # semantica (classificar/reconhecer objetos). Sao complementares, nao
    # concorrentes.
    # ------------------------------------------------------------------
    plt.show()


if __name__ == "__main__":
    main()
