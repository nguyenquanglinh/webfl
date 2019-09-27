import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score


class model_Linear_svc:
    def __init__(self):
        self.model = self.__get_model()

    def convert(self, filename):
        copy = cv2.imread(filename)
        grey_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(grey_img, 90, 255, cv2.THRESH_BINARY_INV)
        contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        s = ""
        for contour in contours:
            (x, y, h, w) = cv2.boundingRect(contour)

            cv2.rectangle(copy, (x, y), (x + h, y + w), (0, 255, 0), 3)
            roi = thresh[y:y + h, x:x + w]
            roi = np.pad(roi, (20, 20), 'constant', constant_values=(0, 0))
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
            nbr = self.model.predict(np.array([roi_hog_fd], np.float32))
            cv2.putText(copy, str(nbr[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            s += str(nbr[0])
            cv2.drawContours(copy, contour, -1, (0,255,0), 3)
        cv2.imwrite("result.jpg", copy)
        return s
        #
        #
        # cv2.imwrite(filename + "_result.jpg", copy)
        # cv2.imshow("abc", copy)
        # cv2.waitKey()

    def __get_model(self):
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # cho x_train
        X_train_feature = []
        for i in range(len(X_train)):
            feature = hog(X_train[i], orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
            X_train_feature.append(feature)
        X_train_feature = np.array(X_train_feature, dtype=np.float32)
        # cho x_test
        X_test_feature = []
        for i in range(len(X_test)):
            feature = hog(X_test[i], orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
            X_test_feature.append(feature)
        X_test_feature = np.array(X_test_feature, dtype=np.float32)
        model = LinearSVC(C=10)
        model.fit(X_train_feature, y_train)
        y_pre = model.predict(X_test_feature)
        self.ti_le_dung = accuracy_score(y_test, y_pre)
        return model
