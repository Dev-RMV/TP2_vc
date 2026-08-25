# -*- coding: utf-8 -*-
"""
ex1b.py — Pipeline de segmentacao de ROI em video (tempo real).

Etapas por frame:
  (1) segmentacao por range HSV (objeto verde)
  (2) limpeza morfologica: erode + dilate (abertura)
  (3) ROI destacada com bounding box + mascara colorida semitransparente
  (4) proporcao area_ROI / area_frame impressa a cada frame

Execucao:  python ex1b.py [--cam]
  --cam : usa a webcam (indice 0) em vez do video sintetico data/bola.mp4
A janela e exibida durante o processamento (q encerra); o video anotado e
frames de evidencia sao gravados em ./outputs em toda execucao.
"""
import argparse
import os

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)

# Range HSV do objeto de interesse (verde). H em OpenCV vai de 0..179.
# Faixa larga em S e V para tolerar sombras/brilho; H restrito (35..85)
# para rejeitar o distrator vermelho e o fundo cinza.
HSV_MIN = np.array([35, 80, 60])
HSV_MAX = np.array([85, 255, 255])

# Kernel eliptico 5x5: erode remove ruido sal-e-pimenta (pixels verdes
# espurios do video), dilate restaura o tamanho do objeto apos a erosao.
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def processa_frame(fr):
    """Retorna (frame anotado, proporcao de area da ROI)."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    masc = cv2.inRange(hsv, HSV_MIN, HSV_MAX)              # (1) segmentacao
    masc = cv2.erode(masc, KERNEL, iterations=2)           # (2) morfologia
    masc = cv2.dilate(masc, KERNEL, iterations=2)

    # (3) ROI: maior contorno da mascara limpa
    contornos, _ = cv2.findContours(masc, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    anotado = fr.copy()
    # overlay semitransparente: pinta a area segmentada de verde e mistura
    overlay = anotado.copy()
    overlay[masc > 0] = (0, 255, 0)
    anotado = cv2.addWeighted(overlay, 0.4, anotado, 0.6, 0)

    proporcao = 0.0
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        x, y, wc, hc = cv2.boundingRect(maior)
        cv2.rectangle(anotado, (x, y), (x + wc, y + hc), (0, 0, 255), 2)
        # (4) proporcao calculada sobre a mascara (area real segmentada),
        # nao sobre o bounding box, que superestimaria objetos nao retangulares
        proporcao = float(np.count_nonzero(masc)) / masc.size
    cv2.putText(anotado, f"area ROI: {proporcao * 100:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return anotado, proporcao


def main(cam: bool):
    fonte = 0 if cam else os.path.join(DATA, "bola.mp4")
    cap = cv2.VideoCapture(fonte)
    assert cap.isOpened(), "Fonte de video indisponivel. Rode prepara_dados.py"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(os.path.join(OUT, "ex1b_saida.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    print("=== ex1b: segmentacao de ROI por HSV ===")
    n = 0
    tempos = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t0 = cv2.getTickCount()
        anotado, prop = processa_frame(fr)
        tempos.append((cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1e3)
        print(f"frame {n:03d}: proporcao ROI = {prop:.4f} ({prop * 100:.2f}%)")
        vw.write(anotado)
        if n in (5, 30, 60):  # frames de evidencia para o relatorio
            cv2.imwrite(os.path.join(OUT, f"ex1b_frame_{n:03d}.png"), anotado)
        cv2.imshow("ROI segmentada (q para sair)", anotado)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        n += 1
    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    # ~2-4 ms/frame em CPU comum -> compativel com tempo real (>30 fps)
    print(f"\n{n} frames | tempo medio de processamento: "
          f"{np.mean(tempos):.2f} ms/frame "
          f"({1000 / np.mean(tempos):.0f} fps teoricos)")
    print("evidencias: outputs/ex1b_saida.mp4 e outputs/ex1b_frame_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", action="store_true",
                    help="usa webcam em vez do video sintetico")
    main(ap.parse_args().cam)
