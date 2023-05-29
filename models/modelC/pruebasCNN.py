
import cv2
from predictionC import  PredictionC

#Clase de prueba que se encarga de leer una imagen e invocar el metodo para predecir su valor.
clases = [6, 7, 8, 9, 10, 11, 12]
ancho=128
alto=128
# miModeloCNN=PredictionC("models/modelC/modeloC.h5",ancho,alto)
miModeloCNN=PredictionC()
# imagen=cv2.imread("dataset/test/10/10_5.jpg")
imagen=cv2.imread("predictions/images/card_0.jpg")
claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()