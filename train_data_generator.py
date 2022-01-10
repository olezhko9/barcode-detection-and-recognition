# Исходный датасет взят с https://www.kaggle.com/karnikakapoor/digits

import os
import numpy as np
from PIL import Image
from cv2 import cv2

base_dir = 'digits_dataset'
x_train = []
y_train = []
for label in sorted(os.listdir(base_dir)):
    for i, img_name in enumerate(sorted(os.listdir(os.path.join(base_dir, label)))):
        if (i + 1) % 4 != 1:  # интересует только каждая 4 картинка
            continue
        img = Image.open(os.path.join(base_dir, label, img_name)).convert('L').resize((28, 28))
        data = np.asarray(img)
        data = cv2.bitwise_not(data)
        x_train.append(data)
        y_train.append(int(label))

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)

shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
y_train = y_train[shuffler]

np.save('train_x.npy', x_train)
np.save('train_y.npy', y_train)
