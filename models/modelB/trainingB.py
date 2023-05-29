import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
##################################

#Carga y prepara las imágenes para el entrenamiento.
def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria+6)+"/"+str(categoria+6)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################
ancho=128
alto=128
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=7

cantidaDatosEntrenamiento=[45,45,45,45,45,45,45]
cantidaDatosPruebas=[15,15,15,15,15,15,15]

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales
model.add(Conv2D(kernel_size=6,strides=3,filters=16,padding="same",activation="tanh",name="capa_1"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=4,strides=2,filters=36,padding="same",activation="tanh",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=2,strides=1,filters=56,padding="same",activation="tanh",name="capa_3"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="tanh"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

#Entrenamiento
model.fit(x=imagenes,y=probabilidades, epochs=40, batch_size=40)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Metricas y matriz de confusion

predicciones = model.predict(imagenesPrueba)
etiquetas_predichas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = np.argmax(probabilidadesPrueba, axis=1)
print("Matriz de Confusión:")
print(confusion_matrix(etiquetas_verdaderas, etiquetas_predichas))
print("Reporte de Clasificación:")
print(classification_report(etiquetas_verdaderas, etiquetas_predichas))

#Matriz de confusion de manera grafica
'''
matriz_confusion = confusion_matrix(etiquetas_verdaderas, etiquetas_predichas)

# Configurar el gráfico
fig, ax = plt.subplots()
im = ax.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Etiquetas de los ejes
ax.set(xticks=np.arange(matriz_confusion.shape[1]),
       yticks=np.arange(matriz_confusion.shape[0]),
       xlabel='Etiquetas Predichas',
       ylabel='Etiquetas Verdaderas')

# Rotular cada celda con el valor de la matriz de confusión
for i in range(matriz_confusion.shape[0]):
    for j in range(matriz_confusion.shape[1]):
        ax.text(j, i, str(matriz_confusion[i, j]),
                ha="center", va="center", color="white")

# Mostrar el gráfico
plt.show()
'''
'''
#cross-validation
scores = cross_val_score(model, imagenes, probabilidades, cv=5,scoring='f1')
print(scores)
print(scores.mean())
'''

# Guardar modelo
ruta="models/modelB/modeloB.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()
