import cv2
import numpy as np

class DetectCards:
    def __init__(self, window) -> None:
        self.name_window = window
        self.increment = 0
    
    def _calculate_areas(self, figuras):
        areas=[]
        for figuraActual in figuras:
            areas.append(cv2.contourArea(figuraActual))
        return areas

    def identify_card(self, gray_image, contours, area):
        areas= self._calculate_areas(contours)
        i = 0
        areaMin = area
        cards = []
        for contour in contours:
            if areas[i]>=areaMin:
                # Crear una imagen en blanco del mismo tamaño que la imagen original
                contour_image = np.zeros_like(gray_image)
                # Dibujar el contorno en la imagen en blanco
                cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)
                # Recortar la región dentro del contorno en la imagen original
                x, y, w, h = cv2.boundingRect(contour)
                cropped = gray_image[y:y+h, x:x+w]
                cards.append(cropped)
            i += 1
        return cards

    def save_card(self, image_card):
        for image in image_card:
            cv2.imwrite(f"predictions/images/card_{self.increment}.jpg", image)
            self.increment += 1


    
