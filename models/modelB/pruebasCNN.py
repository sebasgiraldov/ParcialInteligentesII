
import cv2
from predictionB import  PredictionB

clases = [6, 7, 8, 9, 10, 11, 12]

ancho=128
alto=128

# miModeloCNN=PredictionB("models/modelB/modeloB.h5",ancho,alto)
miModeloCNN=PredictionB()
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