'''
Esse código cria um modelo de CNN, treina-o com o conjunto de dados Fashion MNIST 
e gera a matriz de confusão e a curva ROC para avaliar o desempenho do modelo. 
Em seguida, ele utiliza o Grad-CAM para interpretar o modelo e visualiza o mapa de calor 
resultante em uma imagem de exemplo.
'''

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
         epochs=10, 
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

## ------------------------------------------------------------------------- ##

# Utilizar Grad-CAM para interpretar o modelo

# Selecionar uma imagem de teste
image_index = 0
img = x_test[image_index]
img = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
img = img.reshape(1, 28, 28, 1)

# Obter a camada de saída do modelo
last_conv_layer = model.get_layer('conv2d_2')

# Criar um modelo que retorna tanto a saída quanto a última camada convolucional
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

# Obter os gradientes do modelo em relação à ultima camada convolucional
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, np.argmax([predictions[0]])]

grads = tape.gradient(loss, conv_outputs)[0]

# Calcular os pesos médios dos gradientes
weights = np.mean(grads, axis=(0,1))

# Criar um mapa de calor Grad-CAM
cam = np.dot(conv_outputs[0], weights)

# Redimensionar o mapa de calor para o tamanho da imagem original
cam = cv2.resize(cam, (28, 28))
cam = np.maximum(cam, 0)
cam = cam / np.max(cam)

# Visualizar a imagem original com o mapa de calor Grad-CAM
plt.imshow(img[0, :, :, 0], cmap='gray')
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title('Grad-CAM')
plt.show()

'''
O Grad-CAM (Gradient-weighted Class Activation Mapping) é uma técnica que visa interpretar e visualizar as regiões de uma 
imagem que são mais relevantes para a decisão tomada por um modelo de rede neural convolucional (CNN).

A ideia principal por trás do Grad-CAM é utilizar os gradientes da saída do modelo em relação às ativações da última camada 
convolucional para calcular a importância de cada mapa de ativação na tomada de decisão do modelo. Isso permite gerar um mapa 
de calor que destaca as regiões da imagem que mais influenciaram na classificação.

Aqui está uma visão geral do processo do Grad-CAM:

    Passo 1: Definir a classe de interesse
        No Grad-CAM, é necessário escolher uma classe específica para interpretar. Geralmente, é a classe que o modelo prediz 
        corretamente ou uma classe de interesse específica.

    Passo 2: Obter os gradientes
        Primeiro, é necessário obter os gradientes da saída do modelo em relação às ativações da última camada convolucional. 
        Isso é feito usando a técnica de backpropagation.

    Passo 3: Calcular os pesos dos gradientes
        Em seguida, os gradientes são ponderados calculando a média de cada mapa de ativação ponderado pelo gradiente correspondente. 
        Isso enfatiza as regiões onde os gradientes são mais altos, indicando que essas regiões são mais importantes para a classificação 
        da classe de interesse.

    Passo 4: Gerar o mapa de calor
        O próximo passo é redimensionar os pesos ponderados para o tamanho da imagem original. Normalmente, isso é feito por meio de 
        interpolação ou redimensionamento.

    Passo 5: Visualizar o Grad-CAM
        Por fim, o mapa de calor ponderado é sobreposto na imagem original para destacar as regiões mais importantes. Isso cria uma 
        visualização interpretável que ajuda a entender quais partes da imagem influenciaram a classificação do modelo.

O Grad-CAM é uma técnica eficaz para interpretar modelos de CNN, pois não requer modificações na arquitetura do modelo original. Ele 
pode ser aplicado a diferentes modelos e arquiteturas de CNN, fornecendo insights sobre as regiões de interesse nas imagens e auxiliando 
na compreensão dos processos de tomada de decisão do modelo.

É importante ressaltar que o Grad-CAM interpreta a importância das características visuais aprendidas pelo modelo, mas não leva em 
consideração o contexto global da imagem ou o raciocínio por trás da decisão tomada pelo modelo. Portanto, a interpretação do Grad-CAM 
deve ser complementada com outras técnicas e análises para obter uma compreensão mais completa do funcionamento do modelo.
'''