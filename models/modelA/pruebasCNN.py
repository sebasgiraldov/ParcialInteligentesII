
import cv2
from prediccion import  Prediccion

clases=["numero 0","numero 1","numero 2","numero 3","numero 4","numero 5","numero 6"]

ancho=128
alto=128

miModeloCNN=Prediccion("models/modelA/modeloA.h5",ancho,alto)
imagen=cv2.imread("dataset/test/9/9_0.jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()