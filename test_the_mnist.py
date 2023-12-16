from train import *
import time

model = Model("MyTestModel")

model.add_layer(784)
model.add_layer(32)
model.add_layer(16)
model.add_layer(10)
model.read_weight_file("./weights/%8049.weights")

logger.info("Model configured")

first100_images = []
first100_labels = []
with open("./mnist/t10k-images.idx3-ubyte", "rb") as file:
    test_image_headers = file.read(16)
    for k in range(10000):
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
    for k in range(10000):
        first100_labels.append(integer(file.read(1)))

logger.info("Test labels read")

found_count = 0
counter = 0

begin = time.time()
for image in first100_images:
    v = model.produce(image)
    if v.values.index(maximum(v)) == first100_labels[counter]:
        found_count += 1
    counter += 1
end = time.time()
print(found_count)
