import numpy as np
import cv2
from train import *

model = Model("MyPlaygroundModel")

model.add_layer(784)
model.add_layer(32)
model.add_layer(16)
model.add_layer(10)

model.read_weight_file("./%7902.weights")

logger.info("Model configured")

first100_images = []
first100_labels = []
with open("./mnist/t10k-images.idx3-ubyte", "rb") as file:
    test_image_headers = file.read(16)
    for k in range(100):
        v_list = []
        for x in range(28):
            temp = []
            for y in range(28):
                r = integer(file.read(1))
                temp.append(r)
            v_list.append(Vector(*temp))
        first100_images.append(Matrix(*v_list))

logger.info("Test images read")
with open("./mnist/t10k-labels.idx1-ubyte", "rb") as file:
    test_label_headers = file.read(8)
    for k in range(100):
        first100_labels.append(integer(file.read(1)))

logger.info("Test labels read")



table = np.zeros(shape=(280, 280, 3), dtype=np.uint8)

for a in range(10):
    for b in range(10):
        image = first100_images[10 * a + b]
        for i in range(28):
            for j in range(28):
                val = image[i][j]
                table[28*a + i, 28*b + j] = np.asarray([val, val, val], dtype=np.uint8)


answers = []
for image in first100_images:
    v = model.produce(image)
    answer = v.values.index(maximum(v))
    answers.append(answer)

v = Vector(*answers)
m = v.reshape(10, 10)

print("Models answers are:")
print(m)

cv2.imshow("Test", table)
cv2.waitKey(0)
cv2.destroyAllWindows()