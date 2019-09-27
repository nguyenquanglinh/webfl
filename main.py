import numpy as np
import cv2


def convert_to_binary(img_grayscale, thresh=100):
    thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_binary

# doc du lieu
data = cv2.imread("data.png", 0)
test = cv2.imread("test.jpg", 0)
test = convert_to_binary(test, thresh=100)

# Displaying the image
# cv2.imshow('image', data)
# cv2.waitKey()
# chuyen du lieu ve mang 1 chieu
matrix_data = [np.hsplit(row, 50) for row in np.vsplit(data, 50)]
# cv2.imwrite("test1.jpg",matrix_data[0][0])


x = np.array(matrix_data)

matrix_test = [np.hsplit(row, 1) for row in np.vsplit(test, 1)]
y = np.array(matrix_test)
# lay du lieu train va test
train_data = x[:, :50].reshape(-1, 400).astype(np.float32)
test_data = x[:, 25:50].reshape(-1, 400).astype(np.float32)

test_data_x = y.reshape(-1, 1000000).astype(np.float32)
print(type(test_data_x[0][0]))
# for i in range(len(test_data_x[0])):
#     if (test_data_x[0][i])>150:
#         test_data_x[0][i]=0
#     else:
#         test_data_x[0][i]=255

print(type(test_data_x[0][0]))
# print(test_data_x)
# Gan nhan cho du lieu
k = np.arange(10)
label_data = np.repeat(k, 250)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train_data, 0, label_data)
kq = knn.findNearest(test_data_x, 5)
# print(label_data[500])
print(kq[1])
