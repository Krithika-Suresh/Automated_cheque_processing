# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd='C:\\Users\\Fluffy\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
# # image = cv2.imread("K:\sem_5\DBMS\project\cheque_processing\Cheque_Image_Dataset\cheques\cheque_309091.tif")
# image = cv2.imread("K:\sem_5\DBMS\project\cheque_processing\Cheque_Image_Dataset\cheques\cheque_100828.tif")
# # image = cv2.imread("K:\sem_5\DBMS\project\cheque_processing\cheque_083655.tif")
# image = cv2.resize(image, (960, 540))
# line = image[40:75,700:950]
# gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
# # image = cv2.cvtColor(line,cv2.COLOR_BGR2RGB)
# # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
# # blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# # date_issue = pytesseract.image_to_data(blackhat)
# # print(date_issue)
# cv2.imshow("Image", gray)
# cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation =tf.nn.softmax))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,epochs=3)

accuracy, loss = model.evaluate(x_test, y_test)
# print(accuracy)
# print(loss)

model.save('digits.model')

image = cv2.imread('K:\sem_5\DBMS\project\cheque_processing\Cheque_Image_Dataset\cheques\cheque_100828.tif')
image = cv2.resize(image, (960, 540))
line = image[40:75,700:950]
line = np.invert(np.array([line]))
cv2.imshow("Image",image[40:75,700:950])
cv2.waitKey(0)
prediction = model.predict(line)
print(np.argmax(prediction))