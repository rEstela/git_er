# TESTE 1 - Utilizando biblioteca LIME
# Não Funciona utilizando a biblioteca LIME
# Como o modelo espera receber uma imagem (28, 28, 1) e o LIME só permite usar imagens do tipo (28, 28)
# dá conflito no uso da biblioteca.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Carregar o conjunto de dados Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Pré-processamento dos dados
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Criar o modelo CNN
model = Sequential()
model.add(Conv2D(32, (3 ,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar e treinar o modelo
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']
             )
model.fit(x_train,
         y_train, 
         epochs=5, 
         batch_size=128, 
         validation_data=(x_test, y_test))

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
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de confusão')
plt.show()

# Exibir curva ROC
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='Classe %d (AUC = %0.2f' % (i, roc_auc[i]))
plt.plot([0,1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Usar LIME para interpretar o modelo
explainer = lime_image.LimeImageExplainer()
idx = 0  # Selecionar uma imagem de teste para explicação (pode ser alterado)
explanation = explainer.explain_instance(x_test[idx].reshape(28, 28),
                                         classifier_fn=model.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)

# Plotar a explicação do modelo usando LIME
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.axis('off')
plt.title('LIME Explanation')
plt.show()
