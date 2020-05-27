from skimage.feature.texture import greycomatrix, greycoprops
import process_images as process
import numpy as np
import pandas as pd
import load_images as ld

ImgMatrix = process.get_filtered_images()
OriginalMatrix =ld.load_images()

def get_features(image):
    glcm = greycomatrix(image, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)

    contrast = np.array(greycoprops(glcm, 'contrast'))
    dissimilarity = np.array(greycoprops(glcm, 'dissimilarity'))
    homogeneity = np.array(greycoprops(glcm, 'homogeneity'))
    energy = np.array(greycoprops(glcm, 'energy'))
    correlation = np.array(greycoprops(glcm, 'correlation'))
    ASM = np.array(greycoprops(glcm, 'ASM'))
    listFeatures = [contrast , dissimilarity , homogeneity , energy , correlation ,ASM]
    return listFeatures

FeaturesList = []

for  img in (ImgMatrix):
    listcurrFeatures = get_features(ImgMatrix[img] )
    if("yes" in img):
        listcurrFeatures.append(1)
    else:
        listcurrFeatures.append(0)

    FeaturesList.append(listcurrFeatures)

columns = ["contrast" ,"dissimilarity" , "homogeneity", "energy" , "correlation" , "ASM" , "Label" ]

FeaturesDataset = pd.DataFrame(FeaturesList,columns= columns)

FeaturesListOriginal = []

for  img in (OriginalMatrix):
    listcurrFeatures = get_features(OriginalMatrix[img] )
    if("yes" in img):
        listcurrFeatures.append(1)
    else:
        listcurrFeatures.append(0)

    FeaturesListOriginal.append(listcurrFeatures)

columns = ["contrast" ,"dissimilarity" , "homogeneity", "energy" , "correlation" , "ASM" , "Label" ]

FeaturesDatasetOriginal = pd.DataFrame(FeaturesListOriginal,columns= columns)

def return_DatasetFiltered():
    return FeaturesDataset

def return_DatasetOriginal():
    return FeaturesDatasetOriginal
