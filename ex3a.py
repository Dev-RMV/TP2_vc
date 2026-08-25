# -*- coding: utf-8 -*-
"""
ex3a.py — Comparacao de descritores locais: SIFT x ORB x AKAZE.

Duas imagens da mesma cena sob condicoes diferentes (rotacao 35 graus,
escala 0.75 e iluminacao reduzida — geradas por prepara_dados.py). Para cada
metodo: imagem com keypoints, numero de keypoints, tempo de extracao (ms) e
dimensao/tamanho do descritor por ponto.

Execucao:  python ex3a.py
As tres janelas com keypoints sao exibidas ao final (tecla fecha); as mesmas
imagens sao salvas em ./outputs como evidencia.
"""
import os
import time

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    ref = cv2.imread(os.path.join(DATA, "cena_ref.png"))
    transf = cv2.imread(os.path.join(DATA, "cena_transf.png"))
    assert ref is not None and transf is not None, "Rode prepara_dados.py"
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(transf, cv2.COLOR_BGR2GRAY)

    # SURF nao esta disponivel no opencv-python padrao (patenteado, exige
    # build com OPENCV_ENABLE_NONFREE) -> usamos SIFT, ORB e AKAZE.
    metodos = {
        "SIFT": cv2.SIFT_create(),
        "ORB": cv2.ORB_create(nfeatures=2000),
        "AKAZE": cv2.AKAZE_create(),
    }

    print("=== ex3a: comparacao de descritores ===")
    linhas = []
    for nome, det in metodos.items():
        t0 = time.perf_counter()
        kp1, des1 = det.detectAndCompute(g1, None)
        kp2, des2 = det.detectAndCompute(g2, None)
        dt = (time.perf_counter() - t0) * 1e3 / 2  # media por imagem

        # dimensao do vetor por ponto + tipo (float32 no SIFT = 4 B/dim;
        # binario no ORB/AKAZE = 1 B/dim, comparado por distancia de Hamming)
        dim = des1.shape[1]
        bytes_pp = des1.dtype.itemsize * dim
        linhas.append((nome, len(kp1), len(kp2), dt, dim, bytes_pp,
                       str(des1.dtype)))

        vis = cv2.drawKeypoints(ref, kp1, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(vis, f"{nome}: {len(kp1)} kp", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(OUT, f"ex3a_keypoints_{nome}.png"), vis)
        cv2.imshow(f"keypoints {nome}", vis)

    # ------------------------------------------------------------------
    # TABELA COMPARATIVA (valores MEDIDOS nesta execucao, imagem 512x512):
    #   metodo | kp ref/transf | tempo/img | descritor      | matching
    #   SIFT   |  1105 / 1187  |  ~23 ms   | 128 float32    | L2 (512 B/pt)
    #   ORB    |  2000 / 2000  |  ~50 ms   | 32 uint8 bin.  | Hamming (32 B)
    #   AKAZE  |   870 /  738  |  ~15 ms   | 61 uint8 bin.  | Hamming (61 B)
    # Leitura dos numeros:
    # - SIFT: invariante a escala/rotacao, descritor 16x maior que o ORB
    #   (512 B vs 32 B/ponto) — custo de memoria/matching alto, padrao-ouro
    #   de robustez.
    # - ORB: aqui ficou MAIS LENTO que o SIFT porque nfeatures=2000 força
    #   reter/ordenar muitos FAST corners na piramide de uma imagem pequena
    #   (e o SIFT do OpenCV moderno e vetorizado). Com nfeatures ~500 e o
    #   matching por Hamming incluido na conta, o ORB e o mais barato do
    #   pipeline completo — por isso e o padrao em SLAM embarcado
    #   (ORB-SLAM). Licao: SEMPRE medir no cenario real, nao confiar so na
    #   reputacao do metodo.
    # - AKAZE: mais rapido nesta medicao e com robustez proxima do SIFT
    #   (espaco de escala nao-linear), descritor binario de 61 B — bom
    #   equilibrio, porem detecta menos pontos na imagem transformada
    #   (738 vs 870), sinal de menor repetibilidade sob a mudanca de
    #   iluminacao aplicada.
    # ------------------------------------------------------------------
    print(f"\n{'metodo':7s} {'kp_ref':>7s} {'kp_transf':>9s} "
          f"{'tempo_ms':>9s} {'dim':>4s} {'bytes/pt':>9s} {'tipo':>8s}")
    for nome, k1, k2, dt, dim, bpp, tipo in linhas:
        print(f"{nome:7s} {k1:7d} {k2:9d} {dt:9.1f} {dim:4d} {bpp:9d} "
              f"{tipo:>8s}")
    print("\nevidencias: outputs/ex3a_keypoints_*.png")

    print("pressione tecla em uma janela para sair")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
