from models.modelA.predictionA import PredictionA
from models.modelB.predictionB import PredictionB
from models.modelC.predictionC import PredictionC

# Clase que tiene tres métodos de predicción


class Prediction:

    def __init__(self) -> None:
        self.clases = [6, 7, 8, 9, 10, 11, 12]
        self.predictionA = PredictionA()
        self.predictionB = PredictionB()
        self.predictionC = PredictionC()

    # Toma una lista de cartas de imagen como entrada y devuelve una lista de predicciones del modelo A.
    def prediction_modelA(self, image_cards):
        result_prediction = []
        for card in image_cards:
            result = self.predictionA.predecir(card)
            print('La carta es: ', self.clases[result])
            result_prediction.append(self.clases[result])

        return result_prediction

    # Toma una lista de cartas de imagen como entrada y devuelve una lista de predicciones del modelo B.
    def prediction_modelB(self, image_cards):
        result_prediction = []
        for card in image_cards:
            result = self.predictionB.predecir(card)
            print('La carta es: ', self.clases[result])
            result_prediction.append(self.clases[result])

        return result_prediction
    
    #Toma una lista de cartas de imagen como entrada y devuelve una lista de predicciones del modelo C.
    def prediction_modelC(self, image_cards):
        result_prediction = []
        for card in image_cards:
            result = self.predictionC.predecir(card)
            print('La carta es: ', self.clases[result])
            result_prediction.append(self.clases[result])

        return result_prediction
