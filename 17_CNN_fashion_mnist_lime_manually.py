# TESTE 2 - Gerando LIME manualmente
# Ele mostra os exemplos de perturbações mais próximas da imagem original.
# Isso não é exatamente o que o LIME propõe

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Carregar a base de dados Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Pré-processamento dos dados
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Criar a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar e treinar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Fazer predições
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

# Gerar curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:,i], y_pred[:,i])
    roc_auc[i] = roc_auc_score(test_labels[:,i], y_pred[:,i])

# Plotar a matriz de confusão
plt.imshow(confusion_mtx, cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Plotar a curva ROC
plt.figure()
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Gerar explicação usando LIME (manualmente)
idx = 0  # Selecionar uma imagem de teste para explicação (pode ser alterado)
image = test_images[idx]
true_label = y_true_classes[idx]
pred_label = y_pred[idx]
pred_label = np.argmax(pred_label)

# Definir a função de predição do modelo
def predict_fn(images):
    images = images.reshape(-1, 28, 28, 1)
    return model.predict(images)

# Função para calcular a distância euclidiana entre duas imagens
def euclidean_distance(image1, image2):
    return np.sqrt(np.sum((image1 - image2) ** 2))

# Gerar perturbações aleatórias na imagem de entrada
num_perturbations = 1000
perturbations = np.random.normal(0, 0.1, (num_perturbations, 28, 28, 1))

# Criar cópias da imagem original com as perturbações
images_with_perturbations = np.tile(image, (num_perturbations, 1, 1, 1)) + perturbations

# Calcular as distâncias entre as imagens perturbadas e a imagem original
distances = np.array([euclidean_distance(image, perturbed_image) for perturbed_image in images_with_perturbations])

# Classificar as imagens perturbadas de acordo com suas distâncias
sorted_indices = np.argsort(distances)

# Selecionar as imagens mais próximas da original
num_samples = 5
selected_indices = sorted_indices[:num_samples]

# Plotar as imagens selecionadas
fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
for i, idx in enumerate(selected_indices):
    ax = axes[i]
    ax.imshow(images_with_perturbations[idx].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title('Perturbation {}'.format(i + 1))
plt.suptitle('Images with Perturbations')
plt.show()

# Prever as classes das imagens selecionadas
selected_images = images_with_perturbations[selected_indices]
predictions = predict_fn(selected_images)

# Calcular as probabilidades das classes
probabilities = predictions.mean(axis=0)

# Plotar as probabilidades das classes
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.bar(labels, probabilities)
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Class Probabilities')
plt.show()

# Exibir a classe verdadeira e a classe prevista
print(pred_label)

print('True Label: {}'.format(labels[true_label]))
print('Predicted Label: {}'.format(labels[pred_label]))
