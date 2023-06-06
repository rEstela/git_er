# TESTE 4 - Gerando LIME manualmente
# Com os conhecimentos adquiridos nos testes 1-3, fazer a função LIME manualmente
# Nesse caso, estou gerando um modelo linear resultando em explicações POR PIXEL e não por SEGMENTO


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from skimage.color import gray2rgb

# LIME
import sklearn
import sklearn.metrics
from skimage.segmentation import slic, quickshift, felzenszwalb, watershed
from skimage.color import gray2rgb
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib.patches import Polygon
from skimage import measure


# Carregar base de dados
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessamento dos dados
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test  = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Criar a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2 ,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar e treinar o modelo com EarlyStopping
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=64,
          validation_data=(x_test, y_test), callbacks=[early_stop])

# Fazer predições
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Gerar matriz de confusão
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

# Gerar curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
    roc_auc[i] = roc_auc_score(y_test[:,i], y_pred[:,i])

# Exibir matriz de confusão
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Exibir curva ROC
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='Classe %d (AUC = %0.2f' % (i, roc_auc[i]))
plt.plot([0,1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

###############################################################################################
####### DEFININDO FUNÇÕES PARA LIME #######

def generate_superpixels(image, num_segments = 20, method='slic'):
    """
    Gera superpixels a partir de uma imagem usando um dos métodos de segmentação disponíveis.

    Args:
    - image: numpy array
        A imagem de entrada. Pode ser uma imagem em escala de cinza (2D) ou uma imagem RGB (3D).
    - num_segments: int
        O número desejado de superpixels a serem gerados.
    - method: str (opcional, padrão: 'slic')
        O método de segmentação a ser usado. Pode ser 'slic', 'quickshift' ou 'watershed'.

    Return:
    - segments: numpy array
        Um array contendo a segmentação dos superpixels da imagem. Cada pixel da imagem original é atribuído a um segmento.
    """
    print('-----------------------------')
    print('--> GENERATING SUPERPIXELS')

    if len(image.shape) == 2:
        image = gray2rgb(image)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=2)

    if method == 'slic':
        segments = slic(image, n_segments=num_segments, compactness=10)
    elif method == 'quickshift':
        segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=20)
    elif method == 'watershed':
        gradient = np.gradient(image)
        gradient_magnitude = np.sqrt(np.sum(gradient ** 2, axis=2))
        segments = watershed(gradient_magnitude, markers=num_segments, compactness=0.001)
    else:
        raise ValueError("Invalid segmentation method. Choose from 'slic', 'quickshift', or 'watershed'.")

    print('-> Numero de segmentos: ', len(np.unique(segments)))
    print('-> Segments shape:', segments.shape)
    print('-----------------------------')

    return segments

def generate_perturbations(image, segments, num_samples=100):
    """
    Gera amostras de perturbação com base na imagem e nos segmentos superpixels.

    Args:
    - image: numpy array
        A imagem original.
    - segments: numpy array
        A segmentação dos superpixels da imagem.
    - num_samples: int
        O número de amostras de perturbação a serem geradas.

    Return:
    - perturbations: numpy array
        Um array contendo as imagens de perturbação geradas.
    - flat_perturbations: numpy array
        Um array contendo as imagens (flattened) de perturbação geradas.
    """
    print('-----------------------------')
    print('--> GENERATING PERTURBATIONS')
    perturbations = []
    flat_perturbations = []

    for _ in range(num_samples):
        perturbation = np.random.choice(np.unique(segments))
        mask = (segments == perturbation).reshape(image.shape[:2] + (1,))
        perturb_image = image.copy()
        perturb_image[~mask] = 0
        perturbations.append(perturb_image)
        flat_perturbations.append(perturb_image.flatten())

    
    fig, ax = plt.subplots(2)
    ax[0].imshow(image)
    ax[1].imshow(np.array(perturbations)[0])
    ax[0].set_title('Original Image')
    ax[1].set_title('Perturbation Example')
    plt.show()


    print('-> Perturbations shape: ', np.array(perturbations).shape)
    print('-> Flat perturbations shape: ', np.array(flat_perturbations).shape)
    print('-----------------------------')

    return np.array(perturbations), np.array(flat_perturbations)

def calculate_distances(flat_perturbations, image, distance_type='cosine'):
    """
    Calcula as distâncias entre as features da imagem original e as perturbações geradas.

    Args:
    - image: numpy array
        A imagem original (28x28).
    - perturbations: numpy array
        Uma lista contendo as amostras de perturbação (flatten).
    - distance_type: str (opcional, padrão: 'euclidean')
        O tipo de distância a ser calculada. Pode ser 'euclidean' (euclidiana) ou 'cosine' (cossenos).

    Return:
    - distances: numpy array
        Um array contendo as distâncias calculadas entre a imagem original e as perturbações.
    """
    print('-----------------------------')
    print('CALCULATING DISTANCES')

    distances = sklearn.metrics.pairwise_distances(flat_perturbations, image.flatten()[np.newaxis, ...], metric=distance_type).ravel()
    
    print('-> Distances shape: ', distances.shape)
    print('-----------------------------')
    
    return distances


def fit_linear_model(perturbations, predictions, distances, class_id, kernel_width=0.25, model_type='linear_regression'):
    """
    Ajusta um modelo linear aos dados de treinamento.

    Argumentos:
    - perturbations: numpy array
        As perturbações (flatten) geradas.
    - predictions: numpy array
        As predições do modelo para as perturbações.
    - distances: numpy array
        As distâncias entre a imagem original e as perturbações.
    - kernel_width: float
        A largura do kernel a ser utilizada
    - model_type: str (opcional, padrão: 'linear_regression')
        O tipo de modelo linear a ser ajustado. Pode ser 'linear_regression', 'ridge' ou 'lasso'.

    Retorna:
    - model: objeto do modelo linear
        O modelo linear ajustado aos dados de treinamento.
    """
    print('-----------------------------')
    print('FITTING LINEAR MODEL')
    
    # Kernel
    #weights = np.exp(-distances / kernel_width)
    weights = np.sqrt( np.exp ( - (distances ** 2) / kernel_width ** 2 ) ) #Kernel function

    if model_type == 'linear_regression':
        linear_model = LinearRegression()
    elif model_type == 'ridge':
        linear_model = Ridge()
    elif model_type == 'lasso':
        linear_model = Lasso()
    else:
        raise ValueError("Invalid model type. Choose from 'linear_regression', 'ridge' or 'lasso'.")
      
    linear_model.fit(X=perturbations, y=predictions[:,class_id], sample_weight=weights)

    print('Done.')
    print('-----------------------------')
        
    return linear_model

def visualize_segment_importance(image, segments, weights, num_segments=5):
    """
    Visualiza os segmentos de maior importância em uma imagem.

    Parâmetros:
        - image: A imagem original.
        - segments: Os segmentos gerados pelo algoritmo de segmentação (e.g. SLIC, Quickshift).
        - weights: Os pesos atribuídos a cada segmento.
        - num_segments: O número de segmentos a serem destacados (por ordem de importância).

    Retorna:
        - plot: O plot da imagem original com os segmentos de maior importância destacados.
    """
    # Ordenar os segmentos pelo peso atribuído
    sorted_segments = np.argsort(weights)[::-1]

    # Selecionar os segmentos de maior importância
    top_segments = sorted_segments[:num_segments]

    # Criar uma figura para visualizar a imagem original e os segmentos de maior importância
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')

     # Destacar os segmentos selecionados em verde
    for segment_id in top_segments:
        mask = segments == segment_id

        # Encontrar os contornos do segmento
        contours = measure.find_contours(mask, 0.5)

        # Plotar os contornos encontrados
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='limegreen', linewidth=2)

    ax.set_title('Segmentos de Maior Importância')
    ax.axis('off')

    plt.show()

def keep_top_segments(image, segments, weights, num_segments=5):
    """
    Mantém apenas os N segmentos mais importantes na imagem e define o restante como 0.

    Parâmetros:
        - image: A imagem original.
        - segments: Os segmentos gerados pelo algoritmo SLIC.
        - weights: Os pesos atribuídos a cada segmento.
        - num_segments: O número de segmentos a serem mantidos (por ordem de importância).

    Retorna:
        - new_image: A nova imagem resultante, onde apenas os N segmentos mais importantes são mantidos e o restante é definido como 0.
    """
    # Ordenar os segmentos pelo peso atribuído
    sorted_segments = np.argsort(weights)[::-1]

    # Selecionar os segmentos de maior importância
    top_segments = sorted_segments[:num_segments]

    # Criar uma máscara para manter apenas os segmentos selecionados
    mask = np.isin(segments, top_segments)

    # Aplicar a máscara na imagem para manter apenas os segmentos selecionados
    new_image = np.zeros_like(image)
    new_image[mask] = image[mask]

    plt.figure()
    plt.imshow(new_image, cmap='gray')
    plt.title('LIME - Most important segments')
    plt.show()

    # Criar uma nova imagem com todos os pixels definidos como zero
    new_image_green = np.zeros_like(image)

    # Pintar os segmentos selecionados de verde
    for segment_id in top_segments:
        mask = segments == segment_id
        new_image_green[mask] = 1  # Pintar o segmento de verde

    plt.figure()
    plt.imshow(new_image_green, cmap='gray')
    plt.title('LIME - Binary map')
    plt.show()

    return
#####################################################################################
# RODANDO TUDO
# Selecionar uma imagem de teste para explicação (pode ser alterado)
image_idx = 0
image = x_test[image_idx]

# Nomear cada uma das classes do dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Determinar para qual classe deseja a interpretação
class_pred = model.predict(image[np.newaxis, ...])
class_id = np.argmax(model.predict(image[np.newaxis, ...]))

print('-----------------------------')
print('SELECTING IMAGE TO INTERPRET')
print('-> Image ID: ', image_idx)
#print('-> Model predictions: ', class_pred)
print(f'-> Score: {class_pred[:,class_id]}. Class: {class_names[class_id]}')
print('\n### Interpretando a predição da classe: ', class_names[class_id])
print('-----------------------------')


# Passo 1: Gerar superpixels
segments = generate_superpixels(image, method='quickshift')


# Passo 2: Gerar perturbações
perturbations, flat_perturbations = generate_perturbations(image, segments, num_samples=1000)


# Passo 3: Fazer previsões com o modelo para as perturbações
print('-----------------------------')
print('GETTING PREDICTIONS')
predictions = model.predict(perturbations)
print('-> Predictions shape: ', predictions.shape)
print('-----------------------------')


# Passo 4: Calcular as distâncias entre a imagem original e as perturbações
distances = calculate_distances(flat_perturbations, image)


# Passo 5: Ajustar um modelo linear aos dados e obter os pesos
linear_model = fit_linear_model(flat_perturbations, predictions, distances, class_id)


# Passo 6: Obter os coeficientes do modelo linear que representam 
explanation = linear_model.coef_
print('-> Explanation shape:', explanation.shape)
#print('-> Explanations: ', np.argsort(explanation)[::-1])


# Passo 7: Plotar A imagem original e a importancia de cada segmento com base nos
# coeficintes do modelo linear

# Visualizar os N segmentos de maior importância
visualize_segment_importance(image, segments, explanation, num_segments=250)
# Manter apenas os N segmentos mais importantes
keep_top_segments(image, segments, explanation, num_segments=250)
