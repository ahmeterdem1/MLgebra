# ML with Vectorgebra

A machine learning model on MNIST dataset with only Vectorgebra.

This is a demonstration project on what can be done with Vectorgebra.
The linear algebra basis in Vectorgebra is aimed to be applicable to
larger projects like a machine learning model training. Even though
the code is fully in Python, for projects that are not "company scale",
being fully in Python provides full and easy access to every step and
part of the code/source code even though it is slower. At this scale
the speed may be compromised. And recent updates on Python made it a
lot faster than before. This code works ~2 times faster in Python3.11 
compared to Python3.9. And who knows how fast it will be in the future.

The end goal here is to create toolkits for Python, completely in Python.
Tools being in the same language as the works created with them makes every
step of coding more reachable and understandable by everyone. And in this
examples case, we believe it is a great tool of education. 

Let there be no need for another language if you want so. Yes of course 
writing the code in C++ or something is a lot faster compared to Python, 
but Python is also not too slow. 

## Notes

A successor of this project has been created and currently being developed.
The [MLgebra](https://github.com/ahmeterdem1/ml) library provides easy access
to the algorithms used here with a cleaner control. 

A model with the same structure as studied in this project has been trained to
%84.53 accuracy with the MLgebra library and the weight file can be found here
in the dedicated directory. Codes here aren't updated according to the library,
and left as they were before it.

## Toolkit

The only requirement for this project is [vectorgebra](https://pypi.org/project/vectorgebra/).
However, a general class for machine learning model is created with a subclass of "Node".
Also, the derivative of ReLU is defined to be used in backpropagation.

Files for the MNIST dataset are obviously required for this to work. But those 
are also given here.

For a more human-readable output, a code-part which also displays the
image with the result that is given by the model is commented out. Just
decomment it to use. Numpy and OpenCV is required for that part to work.
Rendering images are done by turning the matrices into numpy arrays and
feeding them to OpenCV.

## Training

This part took a long time to figure out. Firstly we need to go over the model class.

### Model Class

Models are objects with their weights, biases, layers, etc. stored. When
initializing a model, it is just enough to make a call to the init function.
Constructor takes a single argument which defines the name of the model.
This name will later be used to save the weights as a file and will work
as a separator.

After creating the model and giving it a name, one needs to add layers to it.
Initially no layers exist. A layer can be added just by specifying the amount
of neurons it will have. No other distinction is made and specifics are currently
hard coded.

This example is a model of 784x32x16x10 nodes. Each layer in the class consists of
"Node" objects. A node object only has its weights that connects it to the previous
layer as data. So the first layers nodes will have no weights as they are connected
to no other layer coming before them.

After adding the layers, the model needs to be finalized. During finalization, all
weights and biases are generated. Weights are collected into matrices for later use.
Xavier/Glorot initialization is used for this examples weights. Biases are uniformly
generated.

This point is important in the models creation. Weights are generated according to
a well known rule but biases are just uniformly random numbers. The reason is that, 
the input layer has a lot of consistent zero outputs (without the biases) no matter
the image fed. When we think about the MNIST images, we can safely say that the first
and last pixels are _always_ zero. This situation is similar for a lot of other pixels.
Some of them are almost always fully lit, and some of them are almost always fully dim.
This blocks backpropagation flow into the input layer as the situation is consistent
throughout the dataset. To prevent backpropagation from blocking, we introduce uniformly
distributed biases. So the output of an input layer neuron can be literally anything
given the range. At the initial setup at least. Since the backpropagation is not
blocked by any always dim pixels anymore, weights can change according to the learning
of the model.

The training can be done both taking the data one by one or in batches. This example
takes the data one by one. After getting each image, weight and bias update is done.

#### Forward Pass

Input layer is normalized with MinMax normalization, then it is directly passed to the
next layer with its biases. Until the output layer, activator is default ReLU. Softmax
operation is applied to the output layer to calculate the probability distribution.

ReLU, softmax and MinMax are all hard coded. A general ML library will be created later
based on Vectorgebra.

#### Backward Pass

Backward pass is hardcoded to be according to ReLU. Layer by layer, new would-be weights 
are saved into a temporary list. After all layers are done, this operation is repeated
for the biases separately. After only then, Node objects weights get updated. Model
classes weight matrices and bias vectors are also updated.

#### End Result

After all the training is done, user can call the ".save_model()" method to save the weights
in a ".weights" file. The file is named after the models name that is defined during the
initialization.

This file is formatted much alike .csv file format. A line is formatted as below:

`layer_id:node_id:weight1,weight2,...,weightn`

layer_id starts from 0. But keep in mind that first layer in the model does not have
any weights assigned. Assigned weights are to be between the layer and the layer before
that. For the given layers biases, node_id will be "b". 

Also the first line of the file starts with probably -1:b:... This is because
even though the first layer has no weights, it can still have biases. This line represents
the biases of the first layer.

#### Already existing model

With ".read_weight_file(_path_)" an already existing weight file can be copied to a model
that is already finalized. Before doing anything else, you always have to finalize the model.

### Current Example

This example is trained by ".single_train()" function. After each image, weights are updated.
A %72.34 accuracy is reached with learning rate at 0.001, weights initialized with Xavier/Glorot
(Some of the recently added model weight files are originated from the %74 weight file which was
initialized via He initialization, and no modulation was done to biases. It performed better.)
method and biases are uniformly generated between -2 and 2. Generated weights vector gets smaller
hyperbolically as we proceed towards the output layer. This is done because with a lot of testing
it is observed that, weight updates as percentages are a lot smaller in the input layer compared to
the output layer. To achieve a more uniform weight update percentage, bias vectors magnitude is 
lowered hyperbolically towards the output layer. This operation is found to be much beneficial.

There have been a lot of trials with the sigmoid as the activator. Sigmoid needs a cutoff of 5
due to floating point errors (otherwise you even get negative outputs). Using decimal=True 
might help on that. Sigmoid never worked for this example. After the 32-neuron layer, sigmoid 
somehow makes every output identical. It doesn't matter the image provided, outputs are always 
identical. This of course prevents learning. 

At the 16-neuron layer, sigmoid with cutoff results in all strictly binary vector. And as the
inputs to this layer were identical for the whole dataset, all backpropagation is blocked at
this layer. Derivative of the sigmoid is 0 for x ∈ {0, 1}. All delta generated at this layer is
0, and always 0. If you somehow make sigmoid work, please let us know.

#### Updates

Max accuracy is %80.49 for now. It seems like we have reached the local minima. Total weight 
update count to reach this accuracy from the random start is around 1.5 million.

At the latest %80.49 model no overlearning is observed. Performance is almost the same as in 
the training data.

A playground file added that works with numpy and cv2 to display the drawings as well as the
models answers to it. First 100 images are displayed all together in a matrix. Models answers
are done so too.

## Testing

With the weight file provided here, it is tested that the model-%72.34 answers correctly to the
given image. Faulty outputs are observed to be less confident than correct outputs. Standard 
deviation of the output increases when it is incorrect. And sometimes correct answers are also
not confident outputs. 

Learning rate is on the verge of divergence. Still searching for a better convergence.





