import load_images as load
import cv2

ImgMatrix = load.load_images()


filteredImages = dict()

for img in ImgMatrix:
    median = cv2.medianBlur(ImgMatrix[img],3)
    filteredImages[img] = (median)

def get_filtered_images():
    return filteredImages
