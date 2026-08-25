# -*- coding: utf-8 -*-
"""
ex2b.py — Reconhecimento facial com DeepFace (modelo Facenet).

(1) Cadastra 3 identidades a partir de fotos estaticas (data/identidades/,
    montadas do dataset LFW por prepara_dados.py);
(2) Processa o video data/faces_stream.mp4 rotulando cada rosto com o nome
    da identidade ou "desconhecido";
(3) Mede e imprime o tempo medio de inferencia por frame em ms.

DeepFace foi escolhido em vez de face_recognition porque este ultimo depende
do dlib, que exige compilacao com CMake/Visual Studio no Windows; DeepFace
roda sobre o TensorFlow ja instalado para o exercicio 4.

Execucao:  python ex2b.py
O feed anotado e exibido em janela durante o processamento (q encerra);
video anotado e frames de evidencia sao gravados em ./outputs.
"""
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silencia logs verbosos do TF

import cv2
import numpy as np
from deepface import DeepFace

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)

MODELO = "Facenet"          # embedding de 128-d; bom equilibrio precisao/custo
DETECTOR = "opencv"         # detector leve (Haar) -> menor latencia por frame
# Limiar de distancia cosseno para Facenet (valor de referencia da propria
# DeepFace para o par Facenet/cosine). Acima disso -> "desconhecido".
LIMIAR = 0.40


def dist_cosseno(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cadastra_identidades():
    """(1) Gera a galeria: um embedding por foto de cadastro."""
    galeria = []  # lista de (nome, embedding)
    base = os.path.join(DATA, "identidades")
    for pessoa in sorted(os.listdir(base)):
        pasta = os.path.join(base, pessoa)
        for foto in sorted(os.listdir(pasta)):
            rep = DeepFace.represent(
                img_path=os.path.join(pasta, foto), model_name=MODELO,
                detector_backend=DETECTOR, enforce_detection=False)
            galeria.append((pessoa, rep[0]["embedding"]))
        print(f"[cadastro] {pessoa}: "
              f"{len([g for g in galeria if g[0] == pessoa])} fotos")
    return galeria


def identifica(embedding, galeria):
    """Vizinho mais proximo na galeria; rejeita acima do limiar."""
    nome, menor = "desconhecido", float("inf")
    for pessoa, emb in galeria:
        d = dist_cosseno(embedding, emb)
        if d < menor:
            nome, menor = pessoa, d
    return (nome if menor <= LIMIAR else "desconhecido"), menor


def main():
    print("=== ex2b: reconhecimento facial (DeepFace/Facenet) ===")
    galeria = cadastra_identidades()

    cap = cv2.VideoCapture(os.path.join(DATA, "faces_stream.mp4"))
    assert cap.isOpened(), "Rode antes: python prepara_dados.py"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(os.path.join(OUT, "ex2b_saida.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    tempos = []
    n = 0
    contagem = {}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        try:
            # represent com detector: retorna embedding + area facial por rosto
            reps = DeepFace.represent(img_path=fr, model_name=MODELO,
                                      detector_backend=DETECTOR,
                                      enforce_detection=False)
        except Exception:
            reps = []
        # (3) latencia total de inferencia (deteccao + embedding + busca)
        for r in reps:
            fa = r.get("facial_area", {})
            x, y, wf, hf = (fa.get(k, 0) for k in ("x", "y", "w", "h"))
            if wf == 0 or wf >= fr.shape[1] - 2:   # sem deteccao real
                continue
            nome, d = identifica(r["embedding"], galeria)
            contagem[nome] = contagem.get(nome, 0) + 1
            cor = (0, 255, 0) if nome != "desconhecido" else (0, 0, 255)
            cv2.rectangle(fr, (x, y), (x + wf, y + hf), cor, 2)
            cv2.putText(fr, f"{nome} ({d:.2f})", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        dt = (time.perf_counter() - t0) * 1e3
        tempos.append(dt)
        vw.write(fr)
        if n in (10, 60, 120):
            cv2.imwrite(os.path.join(OUT, f"ex2b_frame_{n:03d}.png"), fr)
        cv2.imshow("ex2b (q sai)", fr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n += 1
    cap.release()
    vw.release()
    cv2.destroyAllWindows()

    print(f"\nframes processados: {n}")
    print(f"tempo MEDIO de inferencia por frame: {np.mean(tempos):.1f} ms "
          f"(mediana {np.median(tempos):.1f} ms, "
          f"~{1000 / np.mean(tempos):.1f} fps)")
    print("rotulos atribuidos:", contagem)
    print("evidencias: outputs/ex2b_saida.mp4 e outputs/ex2b_frame_*.png")

    # ------------------------------------------------------------------
    # CUIDADOS ETICOS (comentario exigido no enunciado):
    # Reconhecimento facial embarcado em robos/drones opera sobre pessoas
    # que nao consentiram nem sabem que estao sendo identificadas. Riscos:
    # (1) vigilancia em massa e perseguicao de individuos; (2) vieses de
    # treinamento — taxas de erro maiores para grupos sub-representados
    # levam a falsas identificacoes com consequencias reais (ex. detencoes
    # indevidas); (3) funcao dupla — um drone de inspecao vira instrumento
    # de rastreamento com uma simples troca de software; (4) dados
    # biometricos sao irrevogaveis: um vazamento de embeddings/rostos nao
    # pode ser "trocado" como uma senha. Legislacoes (LGPD, GDPR, AI Act)
    # tratam biometria como dado sensivel: uso exige base legal explicita,
    # minimizacao de dados e avaliacao de impacto.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
