import cv2
import os

folder = 'DataSet'
listOffolder = []
for (root , dirname , filename ) in os.walk(folder):
    for fold in dirname :
        listOffolder.append(os.path.join(root , fold))
images = []
for folder in listOffolder:
    for filename in os.listdir(folder):
        images.append(os.path.join(folder,filename))

imgmatrix = dict()

for i in images:
    img  = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgmatrix[i] = (gray)

def load_images():
    return imgmatrix

def display_images(filename, index):
    cv2.imshow("Image number {}".format(index), filename)
    cv2.waitKey(5000)
