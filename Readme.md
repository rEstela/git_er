# Repositório de Testes de Algoritmos de Machine Learning e Explainable AI

Este repositório contém uma coleção de testes de algoritmos de Machine Learning e técnicas de Explainable AI (Inteligência Artificial Explicável). O objetivo é fornecer exemplos práticos e comparativos de diferentes modelos e métodos de interpretabilidade, a fim de facilitar a compreensão e aplicação dessas técnicas.
Modelos de Machine Learning

Aqui estão os modelos de Machine Learning incluídos neste repositório:

## Logistic Regression (Regressão Logística)

A regressão logística é um modelo de classificação usado para prever a probabilidade de um evento ocorrer. Ele é amplamente utilizado devido à sua simplicidade e interpretabilidade. A regressão logística é adequada para problemas de classificação binária e pode ser estendida para problemas de classificação multiclasse.

## Random Forest (Floresta Aleatória)

A floresta aleatória é um modelo de aprendizado de conjunto que combina várias árvores de decisão. Cada árvore é treinada em uma amostra aleatória dos dados de treinamento e produz uma classificação. A classificação final é determinada pela média das classificações de todas as árvores. As florestas aleatórias são robustas, escaláveis e eficientes em grandes conjuntos de dados.

## XGBoost

O XGBoost (Extreme Gradient Boosting) é uma implementação otimizada de gradient boosting que se destaca pela sua eficácia e velocidade. Ele cria uma sequência de modelos fracos, como árvores de decisão, e combina suas previsões para melhorar o desempenho preditivo. O XGBoost é amplamente utilizado em competições de ciência de dados e em aplicações do mundo real.

## CatBoost

O CatBoost é um algoritmo de gradient boosting que lida naturalmente com dados categóricos, sem a necessidade de codificação manual. Ele possui recursos avançados, como tratamento automático de valores ausentes, suporte a treinamento paralelo e um mecanismo de aprendizado de ranking. O CatBoost é especialmente adequado para conjuntos de dados com variáveis categóricas.

## Naive Bayes (Naïve Bayes)

O Naive Bayes é um modelo probabilístico baseado no teorema de Bayes. Ele assume a independência condicional entre os recursos (daí o "naïve" em seu nome) e é amplamente utilizado para problemas de classificação. O Naive Bayes é rápido, escalável e eficiente em grandes volumes de dados.

## CNNs (Convolutional Neural Networks)

As Redes Neurais Convolucionais (CNNs) são modelos de Deep Learning amplamente usados para tarefas de visão computacional. Elas são especialmente projetadas para processar dados estruturados em grades, como imagens. As CNNs aplicam convoluções e camadas de pooling para extrair características das imagens e usam camadas totalmente conectadas para a classificação.

# Métodos de Interpretabilidade

Neste repositório, você encontrará os seguintes métodos de interpretabilidade:

## Feature Importance (Importância das Características)

A importância das características é um método que quantifica a contribuição de cada variável para as previsões de um modelo. Ele ajuda a identificar as características mais influentes e a entender a relação entre as variáveis de entrada e a saída do modelo.

## GradCAM (Class Activation Mapping baseado em Gradiente)

O GradCAM é uma técnica de explicação visual que destaca as regiões de uma imagem que mais influenciam a decisão de um modelo de CNN. Ele usa gradientes de ativação para mapear as características relevantes na imagem e fornecer interpretabilidade para as previsões da CNN.

## LIME (Local Interpretable Model-agnostic Explanations)

O LIME é um método que gera explicações interpretables para previsões de modelos de Machine Learning. Ele cria modelos locais ao redor de instâncias individuais e analisa como as alterações nas características afetam as previsões. O LIME é um método agnóstico de modelo, o que significa que pode ser aplicado a qualquer algoritmo de Machine Learning.

## SHAP (SHapley Additive exPlanations)

O SHAP é uma abordagem baseada na teoria dos jogos para atribuir valores de importância a cada recurso em uma previsão de modelo. Ele fornece uma explicação individualizada e global para cada instância, destacando a contribuição de cada característica na previsão final. O SHAP é um método versátil e aplicável a uma ampla gama de modelos.