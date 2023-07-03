from PIL import Image
import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.metrics import f1_score, confusion_matrix

train_images_names = [f"0{i}{j}{k}{l}{m}" for i in range(2) for j in range(10) for k in range(10) for l in range(10) for m in range(10)][1:15001]
validation_images_names = [f"0{x}" for x in range(15001, 17001)]
test_images_names = [f"0{x}" for x in range(17001, 22150)]
labels = np.array([int(x) for x in "\n".join(open("train_labels.txt").read().split(",")[1:]).split("\n")[2::2]])

train_images = np.array([np.asarray(Image.open(f"./data/data/{img}.png").convert('L')) for img in train_images_names])
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = labels[:15000]

validation_images = np.array([np.asarray(Image.open(f"./data/data/{img}.png").convert('L')) for img in validation_images_names])
validation_images = validation_images.reshape(validation_images.shape[0], -1)
validation_labels = labels[15000:]

test_images = np.array([np.asarray(Image.open(f"./data/data/{img}.png").convert('L')) for img in test_images_names])
test_images = test_images.reshape(test_images.shape[0], -1)

classifier = GaussianNB()
classifier.fit(train_images, train_labels)
validation_prediction = classifier.predict(validation_images)
validation_score = f1_score(validation_labels, validation_prediction, average='macro')
print(validation_score)

conf_matrix = confusion_matrix(validation_labels, validation_prediction, normalize='pred')
print(conf_matrix)

test_prediction = classifier.predict(test_images)
output = [{'id': img, 'class': prediction} for (img, prediction) in zip(test_images_names, test_prediction)]
with open("submission.csv", "w", newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['id', 'class'])
    writer.writeheader()
    writer.writerows(output)
