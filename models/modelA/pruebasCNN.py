
import cv2
from predictionA import  PredictionA

# clases=["numero 6","numero 7","numero 8","numero 9","numero 10","numero 11","numero 12"]
clases = [6, 7, 8, 9, 10, 11, 12]

ancho=128
alto=128

# miModeloCNN=PredictionA("models/modelA/modeloA.h5",ancho,alto)
miModeloCNN=PredictionA()
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