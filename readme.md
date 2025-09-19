# Desafio de Redes Neurais

## Para cada uma das bases, é necessário treinar um modelo de redes neurais que ultrapasse os seguintes requisitos mínimos:

### Titanic
 - Acurácia >= 78%
 - A base deve ser obtida por meio do seaborn: ```sns.load_dataset("titanic")```
 - Existe um "vazamento de dados" nessa base, a coluna resposta está duplicada, como "alive" e "survived"

### Breast Cancer
 - Acurácia >= 90%
 - A base deve ser obtida por meio do SKLearn: ```sklearn.datasets.load_breast_cancer()```

### Adults
 - Sem requisito mínimo
 - A base está no CSV em anexo

## O que foi feito?

### model.py
Inicialmente, não seria permitido que fosse usado praticamente nada além do Sequential e das camadas Dense, por isso, foi criada a classe "ModelTraining" no arquivo model.py, com o intuito de fazer um processo de otimização que possa ser exectutado em todas as bases.
![Pegando fogo](https://media3.giphy.com/media/v1.Y2lkPTZjMDliOTUyMm9penJuZXM2NTYwNDEzeGdoNDlmNWppZTJoNTFtaWIxenN6eDNkbCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/yr7n0u3qzO9nG/source.gif)
#### 🔥 O processo feito nesse arquivo é meio pesado, principalmente dependendo da quantidade de épocas, já que ele tem um for dentro de um while dentro de um while 🔥
#### A ideia é a seguinte ☝🤓:
Ele sempre vai ter uma camada com um neurônio de saída, com a função sigmoid, pra classificação binária, aí, ele vai criando 1 neurônio por vez e testando, quando ele tiver criado 4 neurônios que não tiveram melhora na performance, ele não os utiliza e cria uma nova camada, repetindo o processo até o modelo não ter aumento de acurácia por 4 camadas.
<br/>
![Neurônios](https://metodosupera.com.br/wp-content/uploads/2014/09/como-funcionam-os-neuronios-exercicios-para-o-cerebro.jpg)
Ao invés de só fazer a verificação >, seria legal poder verificar se o crescimento é significativo, como um teste A/B, porém, imagino que isso não seja possível para esse caso, e não faria sentido fazer com a acurácia do history.

### pre_processamento_bases.py
Aqui, é feita a leitura das bases e o pre processamento, por isso, é criada uma classe que contém os métodos de leitura e preprocessamento.
![Titanic](https://static.nationalgeographicbrasil.com/files/styles/image_3200/public/nationalgeographic762774.jpg?w=1900&h=1268)
