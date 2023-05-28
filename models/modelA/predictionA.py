from keras.models import load_model
import numpy as np
import cv2

class PredictionA():

    def __init__(self) -> None:
        self.modelo=load_model('models/modelA/modeloA.h5')
        self.alto=128
        self.ancho=128

    def predecir(self,imagen):
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas=[]
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA=np.array(imagenesCargadas)
        predicciones=self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=",predicciones)
        clasesMayores=np.argmax(predicciones,axis=1)
        return clasesMayores[0]
