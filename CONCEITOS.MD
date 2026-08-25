# Conceitos Explorados no TP2 — Guia do Zero

Este documento explica, sem assumir conhecimento prévio, todos os conceitos de visão
computacional usados nas 4 tarefas do trabalho. Cada seção indica em qual exercício o
conceito aparece na prática.

---

## 0. O básico: o que é uma imagem para o computador

Uma imagem digital é uma **matriz de números**. Cada posição da matriz é um **pixel**, e
o número guardado nele representa intensidade de luz:

- **Imagem em tons de cinza**: uma matriz única; cada pixel vale de 0 (preto) a 255
  (branco). Uma imagem 640×480 é uma tabela com 480 linhas e 640 colunas.
- **Imagem colorida**: três matrizes empilhadas — os **canais** de cor. O OpenCV usa a
  ordem **BGR** (azul, verde, vermelho); a maioria das outras bibliotecas usa RGB. Cada
  pixel é um trio como (30, 200, 60).
- **Vídeo**: uma sequência de imagens (**frames**) exibidas rapidamente — 30 frames por
  segundo (fps) é o típico. "Processar em tempo real" significa conseguir analisar cada
  frame em menos tempo do que ele demora para chegar (a 30 fps, menos de ~33 ms).

Tudo neste trabalho — profundidade, rostos, keypoints, redes neurais — são operações
matemáticas sobre essas matrizes.

---

## 1. Tarefa 1A — Visão estéreo e profundidade

### O problema
Uma câmera comum "achata" o mundo 3D numa imagem 2D: ela não sabe a que distância as
coisas estão. Robôs e drones precisam dessa distância para desviar de obstáculos.

### A ideia: duas câmeras, como dois olhos
Se você olhar um dedo esticado alternando o olho aberto, o dedo "pula" de posição — e
pula **mais** quanto mais perto ele estiver do rosto. Objetos distantes quase não se
movem. A visão estéreo explora exatamente isso: duas câmeras lado a lado tiram fotos da
mesma cena, e o deslocamento horizontal de cada objeto entre as duas fotos revela sua
distância.

### Disparidade
**Disparidade** é esse deslocamento, medido em pixels: quantas colunas um mesmo ponto da
cena "andou" entre a imagem esquerda e a direita.

- Disparidade **grande** → objeto **perto**
- Disparidade **pequena** → objeto **longe**

A relação exata é `Z = f·B/d`, onde `Z` é a profundidade, `f` a distância focal da
câmera, `B` a **baseline** (distância entre as duas câmeras) e `d` a disparidade. Note
que é uma relação **inversa**: por isso, no ex1a, o pixel de maior disparidade é
declarado "mais próximo". E como `d` está no denominador, quando a disparidade é
minúscula (objeto longe) um erro de 1 pixel muda muito o resultado — o erro de
profundidade cresce com o **quadrado** da distância. É por isso que estéreo com baseline
pequena (drones) só é confiável a poucos metros.

### Mapa de disparidade e o algoritmo SGBM
O **mapa de disparidade** é uma "imagem" onde cada pixel guarda a disparidade daquele
ponto — na prática, um mapa de profundidade relativa da cena inteira. Para calculá-lo, o
algoritmo precisa descobrir, para cada pixel da imagem esquerda, **qual pixel da direita
é o mesmo ponto do mundo** (o problema de *correspondência*). Os dois algoritmos do
OpenCV:

- **StereoBM** (Block Matching): compara pequenos blocos de pixels e escolhe o
  deslocamento em que os blocos ficam mais parecidos. Rápido, mas falha em regiões lisas
  (uma parede branca parece igual em qualquer deslocamento).
- **StereoSGBM** (Semi-Global Block Matching, usado no ex1a): além da semelhança local,
  impõe **suavidade** — vizinhos tendem a ter profundidade parecida — otimizando ao
  longo de várias direções da imagem. Produz mapas mais densos e limpos, custando ~3×
  mais processamento.

Regiões onde não há textura suficiente para casar blocos ficam **inválidas** (os
buracos pretos no mapa gerado). As imagens precisam estar **retificadas** (alinhadas de
modo que pontos correspondentes estejam na mesma linha), o que reduz a busca a 1D.

### Conceitos auxiliares usados
- **Normalização**: reescalar valores para uma faixa padrão (0 a 1) para comparar e
  visualizar — o "0.000 a 1.000" impresso no ex1a.
- **Colormap (COLORMAP_JET)**: converter números em cores (azul = baixo/longe,
  vermelho = alto/perto) para o olho humano interpretar o mapa.

---

## 2. Tarefa 1B — Segmentação por cor e morfologia

### Segmentação
**Segmentar** é separar os pixels que pertencem ao objeto de interesse (a **ROI —
Region of Interest**, região de interesse) dos que não pertencem. A saída é uma
**máscara binária**: uma imagem preto-e-branca onde branco = "é o objeto".

### Por que HSV e não RGB
No espaço BGR/RGB, a cor e a iluminação ficam misturadas nos três canais: um verde na
sombra e um verde no sol têm números completamente diferentes. O espaço **HSV** separa:

- **H (Hue/matiz)**: *qual* cor é — verde, vermelho, azul (no OpenCV, 0 a 179);
- **S (Saturation/saturação)**: quão "viva" ou "lavada" é a cor;
- **V (Value/valor)**: quão clara ou escura.

Para segmentar "coisas verdes", basta exigir H entre ~35 e 85 e aceitar faixas largas de
S e V — o objeto continua sendo encontrado mesmo com sombra ou brilho. É o que o
`cv2.inRange` faz no ex1b: marca de branco todo pixel dentro do intervalo. O quadrado
vermelho distrator do vídeo é rejeitado porque seu H está fora da faixa.

### Morfologia matemática: erosão e dilatação
A máscara crua sai "suja": pixels espúrios que por acaso caíram na faixa de cor (ruído)
e pequenos buracos dentro do objeto. As **operações morfológicas** limpam isso
deslizando um **kernel** (elemento estruturante — no ex1b, uma elipse 5×5) sobre a
máscara:

- **Erosão**: um pixel só permanece branco se **toda** a vizinhança do kernel for
  branca. Efeito: encolhe as regiões brancas e **apaga ruídos isolados** (pontos
  pequenos demais para conter o kernel).
- **Dilatação**: um pixel vira branco se **algum** vizinho no kernel for branco. Efeito:
  expande as regiões brancas e fecha pequenos buracos.

A sequência **erosão → dilatação** chama-se **abertura**: o ruído morre na erosão e não
volta na dilatação, enquanto o objeto grande encolhe e é restaurado ao tamanho original.

### Contornos, bounding box e overlay
- **Contorno** (`findContours`): a linha de fronteira de cada região branca da máscara.
  Pegar o de maior área seleciona o objeto principal.
- **Bounding box** (`boundingRect`): o menor retângulo que envolve o contorno — o
  retângulo vermelho desenhado no vídeo.
- **Overlay semitransparente** (`addWeighted`): misturar duas imagens com pesos
  (40% máscara colorida + 60% imagem original) para destacar a região sem escondê-la.
- **Proporção de área**: nº de pixels brancos da máscara ÷ nº total de pixels do frame.
  No ex1b ela é calculada sobre a máscara, e não sobre o bounding box, porque o
  retângulo incluiria os cantos que não pertencem à bola (superestimando a área).

---

## 3. Tarefa 2A — Detecção de rostos com Haar Cascades

### Detecção ≠ reconhecimento
**Detecção**: "há um rosto aqui?" (encontra a posição). **Reconhecimento**: "de quem é
este rosto?" (atribui identidade). A tarefa 2A faz o primeiro; a 2B, o segundo.

### Como o Haar Cascade funciona
Método clássico (Viola-Jones, 2001), anterior às redes neurais, ainda usado por ser
extremamente leve:

1. **Features de Haar**: padrões retangulares claro/escuro que capturam contrastes
   típicos de um rosto — a região dos olhos é mais escura que a das bochechas, o nariz
   é mais claro que os lados, etc. Cada feature é uma subtração de somas de pixels
   (baratíssima de calcular).
2. **Cascata**: uma janela deslizante percorre a imagem em várias posições e tamanhos.
   Em vez de aplicar todos os testes em toda janela, os classificadores são encadeados
   em estágios: os primeiros, muito simples, descartam rapidamente as janelas
   obviamente sem rosto (a maioria); só as candidatas sobreviventes passam pelos
   estágios caros. Daí o nome *cascade*.
3. O arquivo `haarcascade_frontalface_default.xml` contém os pesos desse classificador
   já treinado com milhares de rostos.

### Os dois parâmetros e o trade-off do exercício
- **scaleFactor**: a janela não sabe o tamanho do rosto, então a imagem é reanalisada em
  vários tamanhos (uma "pirâmide de escalas"). `scaleFactor=1.05` reduz a imagem 5% por
  passo (busca fina, lenta, acha rostos de qualquer tamanho); `1.30` dá saltos de 30%
  (rápido, mas pode "pular" o tamanho exato do rosto e perdê-lo).
- **minNeighbors**: um rosto verdadeiro é detectado várias vezes em janelas vizinhas
  sobrepostas; uma falsa detecção costuma aparecer isolada. Exigir `minNeighbors=6`
  vizinhos filtra os alarmes falsos; aceitar `3` deixa passar mais.

Isso materializa o conceito estatístico central do exercício:

- **Sensibilidade (recall)**: dos rostos que existem, quantos o detector encontra?
- **Especificidade**: das regiões sem rosto, quantas ele corretamente ignora?
- **Falso positivo**: detectar rosto onde não há; **falso negativo**: deixar de detectar
  um rosto real.

Não dá para maximizar os dois ao mesmo tempo: a config sensível (1.05/3) achou 100% dos
rostos mas gerou 31 detecções excedentes e gastou 7,9 ms/frame; a específica (1.30/6)
zerou os excessos e gastou 1,8 ms, perdendo 3,5% dos frames. A escolha certa depende da
aplicação: numa fechadura facial, falso positivo é falha de segurança; num robô que
segue o dono, falso negativo (perder a pessoa) é pior.

---

## 4. Tarefa 2B — Reconhecimento facial com embeddings

### A ideia-chave: transformar rostos em vetores
Comparar fotos pixel a pixel não funciona (iluminação, pose e expressão mudam tudo). Os
sistemas modernos usam uma rede neural para converter cada rosto num **embedding**: um
vetor de números (128 no modelo Facenet) treinado com uma propriedade mágica — **fotos
da mesma pessoa geram vetores próximos; pessoas diferentes, vetores distantes**. A rede
aprendeu isso vendo milhões de trios de fotos (âncora, mesma pessoa, pessoa diferente) e
sendo ajustada para aproximar os dois primeiros e afastar o terceiro.

### O pipeline do ex2b
1. **Cadastro (enrollment)**: para cada identidade, calcular os embeddings de algumas
   fotos e guardá-los numa **galeria** (o "banco de dados de rostos" — 3 pessoas ×
   3 fotos).
2. **Identificação**: para cada frame do vídeo, detectar o rosto, calcular seu embedding
   e procurar o vetor **mais próximo** na galeria (busca por vizinho mais próximo).
3. **Distância cosseno**: a medida de "proximidade" usada — compara o ângulo entre dois
   vetores (0 = idênticos, valores maiores = diferentes).
4. **Limiar de rejeição**: se nem o vizinho mais próximo estiver perto o bastante
   (distância > 0,40), o sistema responde **"desconhecido"** em vez de chutar. Todo
   sistema de reconhecimento sério precisa dessa opção — é melhor não afirmar identidade
   do que afirmar errado. Foi o que aconteceu em 52 dos frames do ex2b: poses muito
   diferentes das fotos de cadastro caíram além do limiar.

### Latência de inferência
**Inferência** é o ato de usar um modelo já treinado para produzir uma resposta. A
latência medida (~128 ms/frame, ~7,8 fps) soma detecção + embedding + busca. Em sistemas
embarcados isso importa: 7,8 fps é aceitável para controle de acesso, mas insuficiente
para rastrear uma pessoa em movimento rápido — daí as otimizações comuns (processar 1 a
cada N frames, modelos menores, aceleradores de hardware).

### Por que a ética entra aqui
Biometria facial identifica pessoas **sem contato e sem consentimento** — diferente de
senha ou digital. Os riscos comentados no código: vigilância em massa; vieses (modelos
erram mais em grupos sub-representados no treinamento, com consequências reais); duplo
uso (um drone de inspeção vira ferramenta de rastreamento só com software); e a
irrevogabilidade — rosto vazado não pode ser "trocado" como uma senha. Por isso
LGPD/GDPR tratam biometria como dado sensível.

---

## 5. Tarefa 3 — Features locais, matching e homografia

### 5.1 Keypoints e descritores (ex3a)

Para localizar-se ou rastrear objetos, um robô precisa encontrar **os mesmos pontos do
mundo em imagens diferentes**. A solução clássica tem duas partes:

- **Keypoint (ponto de interesse)**: um ponto visualmente "marcante" — cantos, quinas,
  manchas de alto contraste — que tende a ser redetectável mesmo se a imagem girar,
  aproximar ou escurecer. (Um pedaço de parede lisa é um péssimo keypoint: parece igual
  em todo lugar.)
- **Descritor**: uma "impressão digital" numérica da vizinhança do keypoint, projetada
  para que o mesmo ponto físico produza descritores parecidos em fotos diferentes.

Os três métodos comparados:

| | SIFT | ORB | AKAZE |
|---|---|---|---|
| Descritor | 128 números reais (512 B) | 32 bytes binários | 61 bytes binários |
| Comparação | distância euclidiana (L2) | distância de Hamming | Hamming |
| Ponto forte | robustez a escala/rotação/iluminação (padrão-ouro) | leveza — padrão em SLAM embarcado | meio-termo: robustez quase-SIFT, custo menor |

- **SIFT** constrói uma pirâmide de escalas e descreve gradientes locais — invariante a
  escala e rotação, mas o descritor é grande e em ponto flutuante (caro para armazenar e
  comparar).
- **ORB** combina um detector de cantos rápido (FAST) com um descritor **binário**
  (BRIEF): cada bit responde "o pixel A é mais claro que o B?". Comparar dois
  descritores vira contar bits diferentes (**distância de Hamming**), que a CPU faz
  quase de graça.
- **AKAZE** usa um espaço de escala não-linear (borra a imagem preservando bordas) com
  descritor binário intermediário.

Lição empírica do exercício: o ORB, famoso por ser o mais rápido, foi o mais **lento**
nesta medição (configuração pediu 2000 features numa imagem pequena) — reforçando que
benchmarks devem ser medidos no cenário real, não assumidos pela reputação.

### 5.2 Matching (ex3b)

**Matching** é parear descritores da imagem A com os da imagem B (candidatos a serem o
mesmo ponto do mundo). Ferramentas usadas:

- **BFMatcher (força bruta)**: compara cada descritor de A com todos os de B. Com
  **cross-check**, só aceita o par se A→B e B→A concordarem (cada um é o melhor vizinho
  do outro) — um filtro de reciprocidade.
- **FLANN**: busca **aproximada** por vizinhos usando estruturas de índice (árvores
  KD) — muito mais rápida quando há milhares de descritores, ao custo de errar
  raramente.
- **Teste de razão de Lowe**: para cada descritor de A, olham-se os **2** melhores
  candidatos em B. Se o 1º não for bem melhor que o 2º (distância < 0,75× a do
  segundo), o match é descartado — porque um ponto ambíguo (textura repetitiva, azulejos,
  janelas de prédio) casa "bem" com vários lugares e não é confiável. No ex3b esse
  filtro reduziu 1105 candidatos a 448 matches bons.

### 5.3 Homografia e RANSAC (ex3b)

- **Homografia**: uma matriz 3×3 que descreve como um **plano** visto de um ângulo se
  transforma quando visto de outro — engloba rotação, escala, translação e perspectiva.
  Com ela, `warpPerspective` "entorta" uma imagem para encaixá-la sobre a outra (o
  alinhamento do exercício).
- **O problema**: mesmo após os filtros, alguns matches estão errados (**outliers**).
  Métodos de ajuste comuns (mínimos quadrados) são arruinados por poucos outliers.
- **RANSAC** (RANdom SAmple Consensus): sorteia repetidamente o mínimo de matches
  necessário (4) para calcular uma homografia candidata e conta quantos dos demais
  matches **concordam** com ela (os **inliers**, com erro de reprojeção < 5 px). A
  candidata com maior consenso vence, e os outliers são simplesmente ignorados.

O **número de inliers** é a métrica prática de confiança: no ex3b, 425 de 448 (94,9%)
— e a homografia recuperada reproduziu numericamente a rotação de 35° e a escala de
0,75 usadas para gerar a imagem, provando que o pipeline inteiro funciona. Em robótica,
esse mesmo pipeline (features → matching → RANSAC → pose) é a base do SLAM: um robô só
"reconhece" um lugar já visitado se a geometria estimada tiver inliers suficientes —
aceitar um falso reconhecimento corromperia o mapa inteiro.

---

## 6. Tarefa 4 — Redes neurais convolucionais (CNNs)

### 6.1 O que é uma rede neural, em uma frase
Uma função matemática gigante com milhões de parâmetros ajustáveis (**pesos**), que
aprende os valores certos vendo exemplos rotulados — em vez de alguém programar as
regras à mão.

### 6.2 Por que "convolucional"
Ligar cada pixel a cada neurônio seria inviável e ignoraria a estrutura espacial da
imagem. A **convolução** resolve: um **filtro** pequeno (3×3 no ex4a) desliza sobre a
imagem calculando, em cada posição, uma soma ponderada da vizinhança. Cada filtro
aprende a detectar **um padrão local** (uma borda vertical, um canto, uma textura), e o
resultado é um **mapa de features** mostrando onde o padrão aparece. Pontos-chave:

- Os pesos do filtro são os mesmos na imagem inteira (**compartilhamento de pesos**):
  um detector de borda serve em qualquer posição — pouquíssimos parâmetros.
- Camadas empilhadas compõem hierarquia: a 1ª camada vê bordas; a 2ª combina bordas em
  formas (bolsos, solas); as densas finais combinam formas em categorias ("tênis").

Componentes da arquitetura do ex4a:

- **ReLU**: função de ativação (`max(0, x)`) aplicada após cada convolução; sem essa
  não-linearidade, empilhar camadas seria matematicamente igual a uma camada só.
- **MaxPooling**: reduz a resolução pegando o máximo de cada janela 2×2. Corta o custo
  em 4× e dá tolerância a pequenos deslocamentos do padrão.
- **Flatten**: achata os mapas 2D num vetor 1D para as camadas seguintes.
- **Dense (totalmente conectada)**: camada clássica em que todo neurônio vê todo o
  vetor; faz a combinação final de evidências.
- **Softmax**: converte os 10 números de saída em probabilidades (positivas, somando 1)
  — "85% tênis, 10% sandália, ...".
- **Dropout**: durante o treino, desliga aleatoriamente 30% dos neurônios a cada passo,
  impedindo a rede de depender de "memorizações" frágeis (regularização).

### 6.3 Como a rede aprende
1. **Loss (perda)**: um número que mede o quão errada está a previsão (aqui,
   *cross-entropy* entre as probabilidades e o rótulo verdadeiro).
2. **Gradiente descendente**: calcula-se em que direção ajustar cada peso para reduzir a
   loss, e dá-se um pequeno passo nessa direção (o otimizador **Adam** adapta o tamanho
   do passo automaticamente).
3. **Época**: uma passada completa pelos 60.000 exemplos de treino; o ex4a fez 10, em
   lotes (**batches**) de 128 imagens por passo.

### 6.4 Generalização e overfitting — o conceito central do item A
Os dados são divididos em três conjuntos com papéis distintos:

- **Treino**: a rede ajusta os pesos com eles;
- **Validação** (10% separado): só para monitorar durante o treino;
- **Teste**: usado uma única vez, no fim, para a nota honesta (91,8% no ex4a).

**Overfitting** é a rede **memorizar** os exemplos de treino em vez de aprender o padrão
geral — como decorar as respostas da lista de exercícios e ir mal na prova. O sintoma
nas curvas: a loss de treino continua caindo enquanto a de **validação** estagna ou
sobe, abrindo um "gap" entre as duas. Foi exatamente o que apareceu (de forma leve) na
época 10 do ex4a. Antídotos: parar cedo (early stopping), mais dados/augmentation, mais
regularização (dropout, L2).

### 6.5 CNN como extratora de features (ex4b)
A última camada (softmax) responde só "qual das 10 classes". Mas a **penúltima** camada
(Dense de 128) contém algo mais valioso: uma **representação** de 128 números que resume
a "essência visual" da imagem — o embedding da imagem (mesmo conceito da seção 4, agora
para objetos em vez de rostos). Cortando a rede nessa camada, ela vira um conversor
imagem → vetor reutilizável para outras tarefas (buscar imagens parecidas, agrupar,
treinar outro classificador em cima) — a base do **transfer learning**.

### 6.6 PCA — enxergando 128 dimensões em 2
Não dá para "plotar" vetores de 128 dimensões. O **PCA** (Análise de Componentes
Principais) encontra as direções de maior variação dos dados e projeta os vetores nas
2 principais — um "achatamento" que preserva o máximo possível da estrutura (59,6% da
variância, no ex4b). Se no gráfico 2D as classes formam **agrupamentos separados** (como
formaram), é evidência de que a rede aprendeu representações **discriminativas**: a
distância no espaço de features passou a significar semelhança semântica. Detalhe
revelador: tênis e sandália ficaram vizinhos — a rede aprendeu sozinha que ambos são
calçados.

### 6.7 O confronto final: descritores clássicos × features aprendidas
A comparação do ex4b fecha o círculo do trabalho inteiro:

| | ORB (tarefa 3) | Features de CNN (tarefa 4) |
|---|---|---|
| O que descreve | vizinhanças de cantos (geometria **local**) | a imagem inteira (semântica **global**) |
| Como é obtido | fórmula fixa, projetada por humanos | **aprendido** dos dados |
| Serve para | casar pontos, estimar pose, SLAM | classificar, reconhecer, buscar por conteúdo |
| Em imagem 28×28 lisa | falha (3 de 20 imagens: zero keypoints) | funciona normalmente |
| "Dois tênis diferentes" | descritores sem relação | vetores próximos |

Não são concorrentes, são **complementares**: o ORB diz *onde* as coisas estão
(geometria); a CNN diz *o que* as coisas são (semântica). Um robô real usa os dois — e é
essa a síntese conceitual do TP2.

---

## 7. Glossário rápido

| Termo | Significado em uma linha |
|---|---|
| Pixel | menor unidade da imagem; um número (cinza) ou trio (cor) |
| ROI | região de interesse — o pedaço da imagem que importa |
| Máscara binária | imagem preto/branco marcando quais pixels pertencem ao objeto |
| Disparidade | deslocamento em pixels de um ponto entre as imagens esquerda/direita; inverso da distância |
| Baseline | distância física entre as duas câmeras de um par estéreo |
| HSV | espaço de cor que separa matiz, saturação e brilho |
| Kernel | matriz pequena que "desliza" sobre a imagem (morfologia, convolução) |
| Erosão / dilatação | encolher / expandir regiões brancas de uma máscara |
| Bounding box | menor retângulo que envolve um objeto detectado |
| Falso positivo / negativo | detectar o que não existe / perder o que existe |
| Sensibilidade (recall) | fração dos casos reais que o sistema encontra |
| Embedding | vetor numérico que representa algo (rosto, imagem) de modo que distância = semelhança |
| Inferência | usar um modelo já treinado para responder; latência = tempo disso |
| Keypoint / descritor | ponto marcante da imagem / impressão digital numérica da sua vizinhança |
| Matching | parear descritores de duas imagens |
| Razão de Lowe | filtro que descarta matches ambíguos comparando os 2 melhores candidatos |
| Homografia | matriz 3×3 que mapeia um plano entre dois pontos de vista |
| RANSAC | ajuste robusto: sorteia amostras mínimas e fica com o modelo de maior consenso |
| Inlier / outlier | dado que concorda / não concorda com o modelo estimado |
| Convolução | filtro deslizante que detecta um padrão local em toda a imagem |
| Pooling | redução de resolução que barateia a rede e dá tolerância a deslocamentos |
| Softmax | transforma saídas da rede em probabilidades que somam 1 |
| Loss | número que mede o erro da rede; treinar = minimizá-lo |
| Época | uma passada completa pelo conjunto de treino |
| Overfitting | memorizar o treino e falhar em dados novos; visível no gap treino×validação |
| Dropout | desligar neurônios aleatórios no treino para evitar overfitting |
| Transfer learning | reaproveitar as features de uma rede treinada em outra tarefa |
| PCA | projeção que reduz dimensões preservando o máximo de variação |
