# TESTE 3 - Gerando LIME manualmente
# Nesse caso só estou gerando imagens perturbadas, vendo qual das perturbações gera melhor ou pior predição,
# Ordeno os segmentos por ordem de valor predito e ploto o segmento que apresentou maior valor predito.
# Não é isso que o LIME propõe ainda

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
from skimage.segmentation import quickshift, mark_boundaries
from skimage.color import gray2rgb

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

model.fit(x_train, y_train, epochs=5, batch_size=64,
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
plt.ylabel('TTrue Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Definir a função de predição do modelo
def predict_fn(images):
    return model.predict(images)

# Função manual do LIME usando Segmentador QuickShift
def lime_quickshift(image, predict_fn):
    # Converter imagem em grayscale para RGB
    rgb_image = gray2rgb(image)
    # Realizar a segmentação
    segments = quickshift(rgb_image, kernel_size=3, max_dist=6, ratio=0.5)
    print('Numero de segmentos: ', len(np.unique(segments)))
    unique_segments = np.unique(segments)

    # Criar imagens de segmentos individuais
    segment_images = []
    for segment_id in unique_segments:
        mask = segments == segment_id
        segment_image = image.copy()
        segment_image[~mask] = 0
        segment_images.append(segment_image[:, :, np.newaxis])
    
    # Predict probabilities for each segment image
    segment_predictions = predict_fn(np.array(segment_images))

    # Calculate the importance score for each segment
    importance_scores = np.max(segment_predictions, axis=1)

    # Select the important segment
    important_segment_idx = np.argmax(importance_scores)
    important_segment_image = segment_images[important_segment_idx]

    return important_segment_image

# Função para plotar a imagem original e o segmento selecionado sobreposto
def plot_segment(image, segment):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.imshow(segment, cmap='Greens', alpha=0.4)
    ax.axis('off')
    plt.show()

# Selecionar uma imagem de teste para explicação (pode ser alterado)
image_idx = 0
image = x_test[image_idx]

# Gerar a imagem do segmento mais importante usando LIME
lime_image = lime_quickshift(image[:,:,0], predict_fn)

# Plotar o segmento mais important
plt.imshow(lime_image.squeeze(), cmap='gray')
plt.axis('off')
plt.title('LIME - Most Important Segment')
plt.show()

# Plotar o resultado
plot_segment(image.squeeze(), lime_image.squeeze())