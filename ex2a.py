# -*- coding: utf-8 -*-
"""
ex2a.py — Deteccao facial em tempo real com Haar Cascades.

Para cada rosto: bounding box + ROI extraida, redimensionada para 48x48 px e
salva numerada em outputs/faces_48/. Duas configuracoes de (scaleFactor,
minNeighbors) sao comparadas quanto a falsos positivos e taxa de deteccao.

Fonte de video: data/faces_stream.mp4 (stream simulado a partir do dataset
LFW, gerado por prepara_dados.py). Cada frame contem exatamente 1 rosto —
isso da um ground truth simples para estimar falsos positivos.

Execucao:  python ex2a.py [--cam]
  --cam : usa a webcam (indice 0) em vez do video simulado. Nesse caso a
          contagem de falsos positivos deixa de valer como metrica (a
          suposicao de 1 rosto/frame e do video sintetico).
O feed e exibido em janela durante o processamento (q encerra); capturas e
ROIs sao salvas em ./outputs em toda execucao.
"""
import argparse
import os

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
FACES = os.path.join(OUT, "faces_48")
os.makedirs(FACES, exist_ok=True)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------------------------------------------------
# TRADE-OFF SENSIBILIDADE x ESPECIFICIDADE (justificativa tecnica):
#
# Config A (scaleFactor=1.05, minNeighbors=3) — ALTA SENSIBILIDADE:
#   piramide de escalas densa (passos de 5%) e exigencia de poucas
#   deteccoes vizinhas para aceitar uma janela. Perde poucos rostos
#   (recall alto), mas aceita mais regioes espurias -> mais falsos
#   positivos e ~3x mais janelas avaliadas (maior latencia, relevante
#   em sistema embarcado).
#
# Config B (scaleFactor=1.30, minNeighbors=6) — ALTA ESPECIFICIDADE:
#   piramide esparsa (passos de 30%) e 6 vizinhos minimos. Quase nao
#   gera falsos positivos e roda mais rapido, mas pode pular a escala
#   em que o rosto aparece -> perde deteccoes (recall menor).
#
# Em controle de acesso embarcado normalmente prefere-se B (falso
# positivo abre porta para intruso); em interacao humano-robo
# prefere-se A (perder o usuario e pior que um box espurio).
# ----------------------------------------------------------------------
CONFIGS = [
    ("A_sensivel", dict(scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))),
    ("B_especifica", dict(scaleFactor=1.30, minNeighbors=6, minSize=(60, 60))),
]


def roda_config(nome, params, fonte):
    cap = cv2.VideoCapture(fonte)
    assert cap.isOpened(), "Fonte indisponivel. Rode antes: python prepara_dados.py"
    n_frame = n_rosto = 0
    frames_com_deteccao = 0
    falsos_pos = 0          # deteccoes alem de 1 por frame (video tem 1 rosto)
    t_total = 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cinza = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        t0 = cv2.getTickCount()
        rostos = CASCADE.detectMultiScale(cinza, **params)
        t_total += (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        if len(rostos) > 0:
            frames_com_deteccao += 1
            falsos_pos += max(0, len(rostos) - 1)
        for (x, y, w, h) in rostos:
            cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # ROI facial -> 48x48 (tamanho padrao de entrada de redes de
            # expressao tipo FER) -> salva numerada
            roi = cv2.resize(cinza[y:y + h, x:x + w], (48, 48))
            cv2.imwrite(os.path.join(FACES, f"{nome}_{n_rosto:04d}.png"), roi)
            n_rosto += 1
        cv2.putText(fr, f"config {nome}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if n_frame == 30:
            cv2.imwrite(os.path.join(OUT, f"ex2a_feed_{nome}.png"), fr)
        cv2.imshow("ex2a (q sai)", fr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n_frame += 1
    cap.release()
    taxa = frames_com_deteccao / n_frame if n_frame else 0.0
    print(f"\nconfig {nome}  {params}")
    print(f"  frames: {n_frame} | taxa de deteccao estimada: {taxa:.1%} "
          f"(frames com >=1 rosto)")
    print(f"  falsos positivos visiveis: {falsos_pos} "
          f"(deteccoes excedentes; video tem exatamente 1 rosto/frame)")
    print(f"  latencia media detectMultiScale: "
          f"{t_total / n_frame * 1e3:.1f} ms/frame")
    print(f"  ROIs 48x48 salvas: {n_rosto}")


def main(cam):
    print("=== ex2a: deteccao facial Haar Cascade ===")
    fonte = 0 if cam else os.path.join(DATA, "faces_stream.mp4")
    if cam:
        print("modo webcam: metricas de falso positivo perdem o ground "
              "truth de 1 rosto/frame do video sintetico")
    for nome, params in CONFIGS:
        roda_config(nome, params, fonte)
    cv2.destroyAllWindows()
    print("\nevidencias: outputs/ex2a_feed_*.png e outputs/faces_48/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", action="store_true",
                    help="usa webcam em vez do video simulado")
    main(ap.parse_args().cam)
