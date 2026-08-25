# -*- coding: utf-8 -*-
"""
ex1a.py — Mapa de disparidade estereo e estimativa de profundidade relativa.

Pipeline: carrega par estereo retificado (aloeL/aloeR, amostra oficial do
OpenCV) -> StereoSGBM -> normalizacao -> COLORMAP_JET -> pixel mais proximo
e mais distante (coordenadas + valor normalizado).

Execucao:  python ex1a.py
As janelas OpenCV sao exibidas ao final; as mesmas visualizacoes tambem sao
salvas em ./outputs como evidencia para o relatorio.
"""
import os

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    esq = cv2.imread(os.path.join(DATA, "aloeL.jpg"))
    dir_ = cv2.imread(os.path.join(DATA, "aloeR.jpg"))
    assert esq is not None and dir_ is not None, "Rode antes: python prepara_dados.py"

    g_esq = cv2.cvtColor(esq, cv2.COLOR_BGR2GRAY)
    g_dir = cv2.cvtColor(dir_, cv2.COLOR_BGR2GRAY)

    # SGBM foi escolhido em vez do BM por produzir mapas mais densos e menos
    # ruidosos em regioes de baixa textura (custo: ~3x mais lento que o BM).
    # numDisparities precisa ser multiplo de 16; blockSize impar.
    block = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,          # alcance de disparidade (px) do par aloe
        blockSize=block,
        P1=8 * 3 * block ** 2,       # penalidades de suavidade recomendadas
        P2=32 * 3 * block ** 2,      # na documentacao do OpenCV
        uniquenessRatio=10,
        speckleWindowSize=100,       # remove "ilhas" de disparidade espuria
        speckleRange=2,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # SGBM devolve disparidade em fixed-point (multiplicada por 16)
    disp = stereo.compute(g_esq, g_dir).astype(np.float32) / 16.0

    # Mascara de pixels validos (disparidade > 0); invalido = sem casamento.
    valida = disp > 0

    # Normalizacao 0..1 apenas sobre pixels validos.
    # Disparidade e INVERSAMENTE proporcional a profundidade (Z = f*B/d):
    #   valor normalizado 1.0 -> disparidade maxima -> pixel MAIS PROXIMO
    #   valor normalizado ~0  -> disparidade minima -> pixel MAIS DISTANTE
    d_min, d_max = disp[valida].min(), disp[valida].max()
    disp_norm = np.zeros_like(disp)
    disp_norm[valida] = (disp[valida] - d_min) / (d_max - d_min)

    # Visualizacao com colormap JET (vermelho = perto, azul = longe)
    disp_vis = cv2.applyColorMap((disp_norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
    disp_vis[~valida] = 0  # pixels sem correspondencia em preto

    # Pixel mais proximo (maior disparidade) e mais distante (menor valida)
    idx_perto = np.unravel_index(np.argmax(np.where(valida, disp, -np.inf)),
                                 disp.shape)
    idx_longe = np.unravel_index(np.argmin(np.where(valida, disp, np.inf)),
                                 disp.shape)
    y_p, x_p = idx_perto
    y_l, x_l = idx_longe
    print("=== ex1a: disparidade e profundidade relativa ===")
    print(f"disparidade valida: min={d_min:.1f}px  max={d_max:.1f}px")
    print(f"pixel MAIS PROXIMO : (x={x_p}, y={y_p})  "
          f"disp={disp[y_p, x_p]:.1f}px  norm={disp_norm[y_p, x_p]:.3f}")
    print(f"pixel MAIS DISTANTE: (x={x_l}, y={y_l})  "
          f"disp={disp[y_l, x_l]:.1f}px  norm={disp_norm[y_l, x_l]:.3f}")

    # marca os dois pixels na visualizacao para evidencia
    cv2.drawMarker(disp_vis, (x_p, y_p), (255, 255, 255),
                   cv2.MARKER_CROSS, 25, 2)
    cv2.putText(disp_vis, "perto", (x_p + 8, y_p - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.drawMarker(disp_vis, (x_l, y_l), (255, 255, 255),
                   cv2.MARKER_TILTED_CROSS, 25, 2)
    cv2.putText(disp_vis, "longe", (x_l + 8, y_l - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(OUT, "ex1a_disparidade_jet.png"), disp_vis)
    cv2.imwrite(os.path.join(OUT, "ex1a_par_estereo.png"),
                np.hstack([esq, dir_]))
    print("evidencias salvas em outputs/ex1a_*.png")

    # ------------------------------------------------------------------
    # COMPORTAMENTO EM CAMERA EMBARCADA EM DRONE (comentario tecnico):
    # 1) Baseline pequeno: drones usam pares estereo com baseline de poucos
    #    cm; como o erro de profundidade cresce com Z^2/(f*B), a estimativa
    #    so e confiavel a poucos metros — alem disso a disparidade cai para
    #    sub-pixel e vira ruido. Por isso drones combinam estereo (perto)
    #    com outros sensores (longe).
    # 2) Vibracao e rolling shutter: a retificacao assume cameras rigidas e
    #    sincronizadas; vibracao do motor desalinha as linhas epipolares e
    #    degrada o matching — exige calibracao robusta e IMU para
    #    compensacao.
    # 3) Custo computacional: SGBM a bordo (CPU ARM) nao roda em tempo real
    #    em resolucao cheia; na pratica usa-se BM/HW dedicado (ex. FPGA) ou
    #    resolucao reduzida, aceitando mapas mais ruidosos.
    # 4) Cenas de voo (ceu, gramado, agua) tem pouca textura -> grandes
    #    regioes invalidas, como as areas pretas do mapa gerado aqui.
    # ------------------------------------------------------------------

    cv2.imshow("par estereo (L | R)", np.hstack([esq, dir_]))
    cv2.imshow("disparidade (COLORMAP_JET)", disp_vis)
    print("pressione qualquer tecla na janela para sair")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
