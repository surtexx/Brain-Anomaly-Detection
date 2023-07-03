import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import cv2
import imutils
import tensorflow as tf
from sklearn.metrics import f1_score
import csv
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])
        topmost = tuple(c[c[:, :, 1].argmin()][0])
        bottommost = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
        return new_image
    else:
        return image


train_images_names = [f"0{i}{j}{k}{l}{m}" for i in range(2) for j in range(10) for k in range(10) for l in range(10)
                      for m in range(10)][1:15001]
validation_images_names = [f"0{x}" for x in range(15001, 17001)]
test_images_names = [f"0{x}" for x in range(17001, 22150)]
labels = np.array([int(x) for x in "\n".join(open("train_labels.txt").read().split(",")[1:]).split("\n")[2::2]])

train_images = []
for img in train_images_names:
    image = cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png", 0)), dsize=(50,50), interpolation=cv2.INTER_CUBIC)/255
    flipped_image = cv2.flip(preprocess_image(image), 1)
    rotated_image1 = rotate_image(preprocess_image(image), 45)
    train_images.append(image)
    train_images.append(flipped_image)
    train_images.append(rotated_image1)

train_images = np.array(train_images, dtype=object)

train_labels = []
for label in labels[:15000]:
    train_labels.append(label)
    train_labels.append(label)
    train_labels.append(label)

train_labels = np.array(train_labels)

validation_images = []
for img in validation_images_names:
    image = cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png", 0)), dsize=(50,50), interpolation=cv2.INTER_CUBIC)/255
    flipped_image = cv2.flip(preprocess_image(image), 1)
    rotated_image1 = rotate_image(preprocess_image(image), 45)
    validation_images.append(image)
    validation_images.append(flipped_image)
    validation_images.append(rotated_image1)

validation_images = np.array(validation_images, dtype=object)

validation_labels = []
for label in labels[15000:]:
    validation_labels.append(label)
    validation_labels.append(label)
    validation_labels.append(label)

validation_labels = np.array(validation_labels)

test_images = []
for img in test_images_names:
    image = cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png", 0)), dsize=(50,50), interpolation=cv2.INTER_CUBIC)/255
    flipped_image = cv2.flip(preprocess_image(image), 1)
    rotated_image1 = rotate_image(preprocess_image(image), 45)
    test_images.append(image)
    test_images.append(flipped_image)
    test_images.append(rotated_image1)

test_images = np.array(test_images, dtype=object)


print("Loaded data.")
nb_train_samples = train_images.shape[0]
nb_validation_samples = validation_images.shape[0]
epochs = 55
batch_size = 70

model = Sequential()

early_callback = EarlyStopping(monitor='accuracy',
                               verbose=1,
                               patience=10,
                               mode='max',
                               restore_best_weights=True)

log_dir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

history = model.fit(
    train_images.astype(np.float32),
    train_labels,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=(validation_images.astype(np.float32), validation_labels),
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[early_callback, tensorboard_callback])

prediction = model.predict(validation_images.astype(np.float32))
prediction = np.round(prediction)
score = f1_score(validation_labels, prediction, average='binary')
print(score)

conf_matrix = confusion_matrix(validation_labels, prediction, normalize='pred')
print(conf_matrix)

test_prediction = model.predict(test_images.astype(np.float32))
test_prediction = np.round(test_prediction)
output = [{'id': img, 'class': int(prediction)} for (img, prediction) in zip(test_images_names, test_prediction[::3])]
with open("submission1.csv", "w", newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['id', 'class'])
    writer.writeheader()
    writer.writerows(output)
