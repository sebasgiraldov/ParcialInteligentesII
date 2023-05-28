import cv2
import numpy as np
from view.detect_cards import DetectCards

class ViewMain:
    KEYS = [55, 56, 57, 48, 107, 106, 113] # [7, 8, 9, 0, k, j, q]

    def __init__(self) -> None:
        self.name_window = 'Parameters'
        self.detection_card = DetectCards(self.name_window)
        self.sum = 0
        self.accumulated = 0
    
    def _nothing(x):
        pass

    def _create_view_main(self):
        cv2.namedWindow(self.name_window)
        cv2.createTrackbar("min", self.name_window, 0, 255, self._nothing)
        cv2.createTrackbar("max",  self.name_window, 100, 255, self._nothing)
        cv2.createTrackbar("kernel", self.name_window, 1, 100, self._nothing)
        cv2.createTrackbar("areaMin", self.name_window, 500, 10000, self._nothing)
    
    def _detect_figure(self, image_original):
        imgame_gris = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        min=cv2.getTrackbarPos("min", self.name_window)
        max=cv2.getTrackbarPos("max", self.name_window)
        binary_image=cv2.Canny(imgame_gris,min,max)
        contours, _=cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaMin = cv2.getTrackbarPos("areaMin", self.name_window)
        for figuraActual in contours:
            message = f'La suma es: {self.sum}'
            message2 = f'El acumulado es: {self.accumulated}'
            cv2.putText(image_original, message, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_original, message2, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawContours(image_original, [figuraActual], 0, (0, 255, 0), 2)
        return imgame_gris, contours, areaMin

    def run_window(self):
        self._create_view_main()
        video = cv2.VideoCapture(1)
        # i = 0
        while True:
            _, frame = video.read()
            imgame_gris, contours, areaMin = self._detect_figure(frame)
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break

            if key == 55:
                image_card = self.detection_card.identify_card(imgame_gris, contours, areaMin)
                self.detection_card.save_card(image_card)

            if key == 112:
                image_card = self.detection_card.identify_card(imgame_gris, contours, areaMin)
                result_prediction = self.detection_card.predict_card(image_card)
                self.sum = result_prediction[0] + result_prediction[1]
                # self.sum = result_prediction[0]
                self.accumulated += self.sum
            
            if key == 99:
                self.sum = 0
                self.accumulated = 0
            
            # if key in self.KEYS:
            #     image_cropped = self.detection_card.identify_card(imgame_gris, contours, areaMin)
            #     cv2.imwrite(f"card_{i}.jpg", image_cropped[0])
            #     cv2.imwrite(f"card_{i}_1.jpg", image_cropped[1])
            #     i += 1
        
        video.release()
        cv2.destroyAllWindows()

    