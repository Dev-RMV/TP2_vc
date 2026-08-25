# -*- coding: utf-8 -*-
"""
ex3b.py — Matching (BFMatcher cross-check + FLANN com razao de Lowe),
homografia por RANSAC e alinhamento por warpPerspective.

Usa o mesmo par de imagens do ex3a (cena_ref / cena_transf) com descritores
SIFT (float -> compativel com FLANN KD-tree e teste de razao de Lowe).

Execucao:  python ex3b.py
As janelas de matches e de alinhamento sao exibidas ao final (tecla fecha);
as mesmas imagens sao salvas em ./outputs como evidencia.
"""
import os

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    ref = cv2.imread(os.path.join(DATA, "cena_ref.png"))
    transf = cv2.imread(os.path.join(DATA, "cena_transf.png"))
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(transf, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    print("=== ex3b: matching + homografia ===")
    print(f"keypoints: ref={len(kp1)}  transf={len(kp2)}")

    # ---- (a) BFMatcher com cross-check: aceita apenas pares que sao
    # mutuamente o melhor vizinho um do outro (filtro simetrico, sem knn).
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    m_bf = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    print(f"BFMatcher (cross-check): {len(m_bf)} matches")

    # ---- (b) FLANN (KD-tree p/ descritores float) + teste de razao de Lowe:
    # um match e confiavel se a distancia ao 1o vizinho for < 0.75x a
    # distancia ao 2o — rejeita descritores ambiguos (texturas repetitivas).
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                  dict(checks=50))
    knn = flann.knnMatch(des1, des2, k=2)
    bons = [m for m, n in knn if m.distance < 0.75 * n.distance]
    print(f"FLANN + razao de Lowe (0.75): {len(bons)} matches bons "
          f"de {len(knn)} candidatos")

    vis = cv2.drawMatches(ref, kp1, transf, kp2, bons[:80], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(OUT, "ex3b_matches.png"), vis)

    # ---- (c) homografia por RANSAC sobre os matches filtrados
    src = np.float32([kp1[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, masc = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(masc.sum())
    print(f"RANSAC: {inliers} inliers de {len(bons)} matches "
          f"({inliers / len(bons):.1%})")
    print("homografia estimada:\n", np.round(H, 4))

    # ---- (d) alinhamento: projeta a referencia sobre a imagem transformada
    h, w = transf.shape[:2]
    alinhada = cv2.warpPerspective(ref, H, (w, h))
    # sobreposicao 50/50 para evidenciar visualmente o alinhamento
    sobre = cv2.addWeighted(alinhada, 0.5, transf, 0.5, 0)
    cv2.imwrite(os.path.join(OUT, "ex3b_alinhada.png"), alinhada)
    cv2.imwrite(os.path.join(OUT, "ex3b_sobreposicao.png"), sobre)
    print("evidencias: outputs/ex3b_matches.png, ex3b_alinhada.png, "
          "ex3b_sobreposicao.png")

    # ------------------------------------------------------------------
    # RELEVANCIA PARA LOCALIZACAO VISUAL EM ROBOTICA (comentario tecnico):
    # O numero de inliers do RANSAC e a metrica pratica de confianca do
    # registro: outliers (matches errados) sao inevitaveis, e o RANSAC
    # encontra o modelo geometrico consistente com o maior consenso.
    # Em SLAM/relocalizacao, um lugar so e considerado "reconhecido"
    # quando a homografia/pose estimada tem inliers suficientes (tipico
    # >15-30); poucos inliers indicam falso loop-closure, que corromperia
    # o mapa. O mesmo pipeline (features -> matching -> RANSAC -> pose)
    # e a base de rastreamento de marcadores, servoing visual e
    # estabilizacao de imagem em drones.
    # ------------------------------------------------------------------

    cv2.imshow("matches (FLANN+Lowe)", vis)
    cv2.imshow("ref alinhada sobre transf", sobre)
    print("pressione tecla em uma janela para sair")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
