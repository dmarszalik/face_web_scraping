import cv2 as cv
import numpy as np
import requests
import config as c
import os


class FaceDataScraper:
    def __init__(self, destination_path, accepted_gender=None, accepted_age=None, n_images=15, model_mean_values=[104, 117, 123]):
        self.destination_path = destination_path
        self.accepted_gender = accepted_gender
        self.accepted_age = accepted_age or ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
        self.n_images = n_images
        self.model_mean_values = model_mean_values

    def check_gender(self, image):
        genderList = ['Male', 'Female']
        blob = cv.dnn.blobFromImage(image, 1, (227, 227), self.model_mean_values, swapRB=False)

        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        genderNet = cv.dnn.readNet(genderModel, genderProto)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        return gender

    def check_age(self, image):
        ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
        blob = cv.dnn.blobFromImage(image, 1, (227, 227), self.model_mean_values, swapRB=False)

        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        ageNet = cv.dnn.readNet(ageModel, ageProto)

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        return age

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return np.frombuffer(response.content, np.uint8)
        else:
            print(f"Nie udało się pobrać obrazu z {url}")
            return None

    def scrap_images(self):
        n = 1
        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)
        for i in range(self.n_images):
            image_bytes = self.download_image(c.SOURCE_URL)
            if image_bytes is not None:
                image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)

                recognized_gender = self.check_gender(image)
                age = self.check_age(image)

                if (self.accepted_gender is None or recognized_gender == self.accepted_gender) and age in self.accepted_age:
                    with open(rf"{self.destination_path}/file_{n}.jpeg", "wb") as f:
                        f.write(image_bytes)
                    n += 1


# Create an instance of the class and start scraping images
scraper = FaceDataScraper(c.DESTINATION_PATH, c.ACCEPTED_GENDER, c.ACCEPTED_AGE, c.N_IMAGES, c.MODEL_MEAN_VALUES)
scraper.scrap_images()