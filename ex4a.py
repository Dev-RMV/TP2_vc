# -*- coding: utf-8 -*-
"""
ex4a.py — CNN do zero para classificacao do Fashion-MNIST (10 classes).

Conforme instrucoes do trabalho: dataset Fashion-MNIST e Keras rodando com
BACKEND TENSORFLOW (definido explicitamente via KERAS_BACKEND antes do
import). Arquitetura: 2 blocos Conv2D+MaxPooling -> Flatten -> Dense ->
softmax. Treino de 10 epocas com curvas de acuracia/loss e acuracia de teste.

Execucao:  python ex4a.py
As curvas de treino sao exibidas em janela ao final e tambem salvas em
./outputs como evidencia.
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"   # Keras 3 sobre TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "outputs")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["camiseta", "calca", "pulover", "vestido", "casaco",
           "sandalia", "camisa", "tenis", "bolsa", "bota"]


def constroi_modelo():
    """2 blocos conv (32 e 64 filtros) + densa de 128.
    Justificativas: filtros 3x3 (padrao moderno, campo receptivo cresce com
    profundidade a custo baixo); dobrar filtros apos pooling mantem
    capacidade enquanto a resolucao espacial cai; Dropout de 0.3 na cabeca
    densa combate overfitting, ja que Fashion-MNIST e pequeno (60k, 28x28)."""
    return keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),                       # bloco 1
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),                       # bloco 2
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu", name="features"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ], name="cnn_fmnist")


def main():
    print("=== ex4a: CNN Fashion-MNIST (backend:", keras.backend.backend(), ") ===")
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
    # normalizacao 0..1 e canal explicito (28,28,1)
    x_tr = (x_tr / 255.0).astype("float32")[..., None]
    x_te = (x_te / 255.0).astype("float32")[..., None]

    modelo = constroi_modelo()
    modelo.summary()
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    # 10% do treino separado para validacao -> curvas treino vs. validacao
    hist = modelo.fit(x_tr, y_tr, epochs=10, batch_size=128,
                      validation_split=0.1, verbose=2)

    perda, acc = modelo.evaluate(x_te, y_te, verbose=0)
    print(f"\nacuracia no conjunto de TESTE: {acc:.4f}  (loss {perda:.4f})")

    # curvas de treino vs. validacao
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, 11)
    a1.plot(ep, hist.history["accuracy"], "o-", label="treino")
    a1.plot(ep, hist.history["val_accuracy"], "s-", label="validacao")
    a1.set(title="Acuracia", xlabel="epoca", ylabel="acuracia")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, hist.history["loss"], "o-", label="treino")
    a2.plot(ep, hist.history["val_loss"], "s-", label="validacao")
    a2.set(title="Loss", xlabel="epoca", ylabel="loss")
    a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle("ex4a — CNN Fashion-MNIST: treino vs. validacao")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ex4a_curvas.png"), dpi=120)
    print("grafico salvo em outputs/ex4a_curvas.png")

    # diagnostico objetivo de overfitting: gap entre treino e validacao
    gap_acc = hist.history["accuracy"][-1] - hist.history["val_accuracy"][-1]
    gap_loss = hist.history["val_loss"][-1] - hist.history["loss"][-1]
    print(f"gap final treino-validacao: acc={gap_acc:+.4f}  "
          f"loss={gap_loss:+.4f}")
    # ------------------------------------------------------------------
    # OVERFITTING (comentario): ha overfitting quando a loss de validacao
    # para de cair (ou sobe) enquanto a de treino continua caindo, abrindo
    # o "gap" entre as curvas. Nesta arquitetura, com Dropout 0.3 e apenas
    # 10 epocas, o gap tipico observado e pequeno (acc de treino poucos
    # pontos acima da validacao e val_loss estabilizando nas ultimas
    # epocas) — indicio de overfitting LEVE/INCIPIENTE: o modelo comeca a
    # memorizar, mas ainda generaliza bem (ver numeros impressos acima,
    # que sao o criterio objetivo). Mitigacoes se fosse treinar mais:
    # early stopping, data augmentation ou mais dropout/L2.
    # ------------------------------------------------------------------

    modelo.save(os.path.join(OUT, "modelo_fmnist.keras"))  # usado pelo ex4b
    print("modelo salvo em outputs/modelo_fmnist.keras")
    plt.show()


if __name__ == "__main__":
    main()
