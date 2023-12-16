from vectorgebra import *
import os

dec = False

def deriv_relu(x):
    # Replaced o * (1 - o)
    # return 1 if x >= cutoff else 0.1
    return 1 if x >= 0 else 0

class PathError(Exception):
    def __init__(self):
        super().__init__("Incorrect path or filename.")

class Node:

    def __init__(self):
        self.w = None

class Model:

    def __init__(self, name: str = ""):
        if os.path.exists(f"./{name}.weights"):
            raise PathError()
        self.name = name
        self.layers = []
        self.errors = []
        self.bias = []
        self.last_output = []
        self.w_matrices = []

    def add_layer(self, amount: int = 1):
        self.layers.append([Node() for k in range(amount)])

    def save_model(self):
        with open(f"{self.name}.weights", "x") as file:
            layer_count = 0
            file.write(f"-1:b:" + ",".join([str(k) for k in self.bias[0]]) + "\n")
            for layer in self.layers[1:]:
                node_count = 0
                for node in layer:
                    file.write(f"{layer_count}:{node_count}:" + ",".join([str(k) for k in node.w.values]) + "\n")
                    node_count += 1
                file.write(f"{layer_count}:b:" + ",".join([str(k) for k in self.bias[layer_count + 1]]) + "\n")
                layer_count += 1

    def read_weight_file(self, path: str = ""):
        if not path.endswith(".weights"): raise PathError()
        with open(path, "r") as file:
            all = file.read()

        lines = all.split("\n")
        for line in lines:
            if line == "":
                continue
            parts = line.split(":")
            i = int(parts[0])

            if parts[1] == "b":
                # "i" is not important here.
                weights = parts[2].split(",")
                self.bias.append(Vector(*[float(k) for k in weights]))
            else:
                j = int(parts[1])
                weights = parts[2].split(",")

                self.layers[i + 1][j].w = Vector(*[float(k) for k in weights])

        for layer in self.layers[1:]:
            w_list = []
            for node in layer:
                w_list.append(node.w)
            self.w_matrices.append(Matrix(*w_list))

    def finalize(self):
        for i in range(1, len(self.layers)):
            w_list = []
            for node in self.layers[i]:
                #node.w = Vector.randVfloat(dim=len(self.layers[i - 1]), a=-4 + i, b=4 - i, decimal=False)
                node.w = Vector.randVgauss(dim=len(self.layers[i - 1]), mu=0, sigma=(1/sqrt(len(self.layers[i - 1]))), decimal=dec)
                w_list.append(node.w)
            self.w_matrices.append(Matrix(*w_list))

        for i in range(0, len(self.layers)):
            self.bias.append(Vector.randVfloat(dim=len(self.layers[i]), a=-2, b=2, decimal=dec) / (i + 1))

    def update_matrices(self):
        self.w_matrices = []
        for layer in self.layers[1:]:
            temp = []
            for node in layer:
                temp.append(node.w)
            self.w_matrices.append(Matrix(*temp))

    def single_train(self, d, label, learning_rate):
        self.last_output = [Vector.zero(len(k), decimal=dec) for k in self.layers]
        error_list = []
        error = Vector.zero(len(self.layers[-1]), decimal=dec)

        if isinstance(d, Matrix):
            dims = len(d.values) * len(d.values[0])
            temp = d.reshape(dims)
        else:
            temp = d

        temp = temp.minmax()
        # Normalization is done before adding the bias.
        temp += self.bias[0]

        counter = 1
        self.last_output[0] += temp
        for matrix in self.w_matrices:

            if self.w_matrices.index(matrix) == 0:
                # We have already applied the minmax. So no sigmoid.
                temp = matrix * temp
                temp += self.bias[counter]

            elif self.w_matrices.index(matrix) == len(self.w_matrices) - 1:
                temp = matrix * temp
                temp += self.bias[counter]
                temp = temp.softmax()
            else:
                temp = matrix * temp
                temp += self.bias[counter]
                temp = temp.relu()
            self.last_output[counter] += temp
            counter += 1

        error += label - temp
        error_list.append(error)

        previous_delta = False
        new_weights = []
        new_biases = []
        for index in range(len(self.layers) - 1, 0, -1):
            list_for_prev_delta = []
            new_weights.insert(0, [])
            for node_index in range(len(self.layers[index])):
                node = self.layers[index][node_index]
                o = self.last_output[index][node_index]

                if not previous_delta:
                    delta = error_list[-1][node_index]
                    list_for_prev_delta.append(delta)
                    new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                else:
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[index + 1]:
                        error_sum += prev_node.w[node_index] * error_list[-1][counter]
                        counter += 1

                    delta = deriv_relu(o) * error_sum
                    list_for_prev_delta.append(delta)
                    new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
            error_list.append(Vector(*list_for_prev_delta))
            previous_delta = True

        last_delta = []
        for i in range(len(self.layers[0])):
            o = self.last_output[0][i]
            error_sum = 0
            counter = 0
            for prev_node in self.layers[1]:
                error_sum += prev_node.w[i] * error_list[-1][counter]
                counter += 1
            last_delta.append(deriv_relu(o) * error_sum)
        error_list.append(Vector(*last_delta))

        for i in range(len(self.layers)):
            new_biases.append(self.bias[i] + learning_rate * error_list[-1 - i])
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].w = new_weights[i - 1][j]
        self.bias = [v for v in new_biases]
        self.update_matrices()

    def train(self, data, labels, learning_rate):
        self.last_output = [Vector.zero(len(k), decimal=False) for k in self.layers]
        error_list = []
        error = Vector.zero(len(self.layers[-1]), decimal=False)
        label_id = 0

        for d in data:
            if isinstance(d, Matrix):
                dims = len(d.values) * len(d.values[0])
                temp = d.reshape(dims)
            else:
                temp = d

            temp += self.bias[0]
            temp = temp.minmax()

            counter = 0
            self.last_output[counter] += temp
            for matrix in self.w_matrices:
                if counter != 0:
                    temp += self.bias[counter]
                temp = matrix * temp
                temp = temp.sig(cutoff=5)
                if counter != len(self.w_matrices) - 1:

                    self.last_output[counter + 1] += temp
                else:
                    # Softmax operation
                    temp = temp.softmax()
                    self.last_output[counter + 1] += temp
                counter += 1

            error += labels[label_id] - temp
            label_id += 1

        for i in range(len(self.last_output)):
            self.last_output[i] /= len(data)
        error_list.append(error / len(data))
        previous_delta = False
        new_weights = []
        new_biases = []
        for index in range(len(self.layers) - 1, 0, -1):
            list_for_prev_delta = []
            new_weights.insert(0, [])
            for node_index in range(len(self.layers[index])):
                node = self.layers[index][node_index]
                o = self.last_output[index][node_index]

                if not previous_delta:
                    delta = o * (1 - o) * error_list[-1][node_index]
                    list_for_prev_delta.append(delta)
                    new_weights[0].append(node.w - learning_rate * delta * self.last_output[index - 1])
                else:
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[index + 1]:
                        error_sum += prev_node.w[node_index] * error_list[-1][counter]
                        counter += 1

                    delta = o * (1 - o) * error_sum
                    list_for_prev_delta.append(delta)
                    new_weights[0].append(node.w - learning_rate * delta * self.last_output[index - 1])
            error_list.append(Vector(*list_for_prev_delta))
            previous_delta = True

        last_delta = []
        for i in range(len(self.layers[0])):
            o = self.last_output[0][i]
            error_sum = 0
            counter = 0
            for prev_node in self.layers[1]:
                error_sum += prev_node.w[i] * error_list[-1][counter]
                counter += 1
            last_delta.append(o * (1 - o) * error_sum)
        error_list.append(Vector(*last_delta))


        for i in range(len(self.layers)):
            new_biases.append(self.bias[i] + learning_rate * error_list[-1 - i])
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].w = new_weights[i - 1][j]
        self.bias = [v for v in new_biases]
        self.update_matrices()

    def produce(self, d):
        if isinstance(d, Matrix):
            dims = len(d.values) * len(d.values[0])
            temp = d.reshape(dims)
        else:
            temp = d

        temp = temp.minmax()
        temp += self.bias[0]
        counter = 1
        for matrix in self.w_matrices:
            if self.w_matrices.index(matrix) == 0:
                # We have already applied the minmax. So no sigmoid.
                temp = matrix * temp
                temp += self.bias[counter]
            elif self.w_matrices.index(matrix) == len(self.w_matrices) - 1:
                temp = matrix * temp
                temp += self.bias[counter]
                temp = temp.softmax()
            else:
                temp = matrix * temp
                temp += self.bias[counter]
                temp = temp.relu()
            counter += 1
        return temp

all_partitions = []

def integer(b):
    return int.from_bytes(b, byteorder="big")

if __name__ == "__main__":

    with open("./mnist/train-images.idx3-ubyte", "rb") as training_set:
        image_header = training_set.read(16)
        for k in range(60000):
            all_images = []
            for count in range(1):
                # ALl images are processed in this loop.
                temp_image = []
                for k in range(28):
                    temp = []
                    for l in range(28):
                        r = integer(training_set.read(1))
                        temp.append(r)
                    temp_image.append(Vector(*temp))

                all_images.append(Matrix(*temp_image))

            # Appends a list of matrices
            all_partitions.append(all_images)

    logger.info("Dataset read")
    label_partitions = []

    with open("./mnist/train-labels.idx1-ubyte", "rb") as training_label:
        label_header = training_label.read(8)
        for k in range(60000):
            partition = []
            for l in range(1):
                v = Vector.zero(10, decimal=dec)
                i = integer(training_label.read(1))
                v[i] = 1
                partition.append(v)
            label_partitions.append(partition)

    logger.info("Labels read")
    """
    When we reach here, all the images are read and divided into partitions.
    After each partition, weights will get updated. In this example, partitions
    consist of 100 images each. There are therefore 600 partitions.
    
    Their labels are also read and structured accordingly.
    
    """

    model = Model("MyLastModel2")
    model.add_layer(784)
    model.add_layer(32)
    model.add_layer(16)
    model.add_layer(10)
    #model.finalize()
    model.read_weight_file("./weights/%8049.weights")
    logger.info("Model finalized, training is starting.")
    lrate = 0.00000009
    for i in range(60000):
        model.single_train(all_partitions[i][0], label_partitions[i][0], lrate)
        if i % 1000 == 0:
            print(f"%{(i / 600):.2f}")
    logger.info("Training finished")

    model.save_model()
    logger.info("Model saved")
