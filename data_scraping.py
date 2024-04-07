import cv2 as cv
import numpy as np
import requests
import config as c
import os

def check_gender(genderNet, image, model_mean_values=[104, 117, 123]):
    genderList = ['Male', 'Female']
    blob = cv.dnn.blobFromImage(image, 1, (227, 227), model_mean_values, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    return gender

def check_age(ageNet, image, model_mean_values=[104, 117, 123]):
    ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
    blob = cv.dnn.blobFromImage(image, 1, (227, 227), model_mean_values, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    return age

def scrap_images(destination_path, gender=None, accepted_age=['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)'], n_images=15, model_mean_vaules=[104, 117, 123]):
    n = 1
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for i in range(n_images):
        person = requests.get(c.SOURCE_URL, headers={'User-Agent': 'My User Agent 1.0'}).content

        image_bytes = np.frombuffer(person, np.uint8)
        image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)

        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        genderNet = cv.dnn.readNet(genderModel, genderProto)

        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        ageNet = cv.dnn.readNet(ageModel, ageProto)

        recognized_gender = check_gender(genderNet, image, model_mean_vaules)
        age = check_age(ageNet, image)

        if (recognized_gender == gender and age in accepted_age):
            with open(rf"{destination_path}/file_{n}.jpeg", "wb") as f:
                f.write(person)
            n += 1
        elif (gender is None and age in accepted_age):
            with open(rf"{destination_path}/file_{n}.jpeg", "wb") as f:
                f.write(person)
            n += 1
        else:
            pass



scrap_images(c.DESTINATION_PATH, gender=c.ACCEPTED_GENDER, accepted_age=c.ACCEPTED_AGE, n_images=c.N_IMAGES, model_mean_vaules=c.MODEL_MEAN_VALUES)