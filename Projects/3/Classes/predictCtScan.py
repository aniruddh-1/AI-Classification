############################################################################################
#
# Project:       Peter Moss COVID-19 AI Research Project
# Repository:    COVID-19 AI Classification
# Project:       COVID-19 Pneumonia Detection/Early Detection
#
# Author:        Aniruddh Sharma
# Title:         Predict CT Scan
# Description:   Analyze the CT Scan images and predict whether they are COVID-19 or normal Scans by using Pretrained Model
# License:       MIT License
# Last Modified: 2020-06-09
#
############################################################################################


import cv2
import tensorflow as tf
categories = ["covid19_scan","normal_scan"]

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model('covid19_pneumonia_detection_cnn.h5')	 #provides the path of your trained CNN model
prediction = model.predict([prepare(' ')])                              #paste the PNG image in Classes Directory and write the name of image file in inverted colon like for covid_scan image file - 'covid_scan.png'

print(categories[int(prediction[0][0])])