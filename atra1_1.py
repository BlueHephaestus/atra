"""
ATRA MK1.1 - Affine Transformation Autoencoders

This model is only for starters. 
This method is really complex and unordinary, so we will have to build many of our own parts out of symbolic math operations,
    and use pre-existing layers wherever applicable.
Because of the complexity of it, this method will only use a mini batch size of 1, and won't do transformations.
This is just to ensure we can properly train the autoencoder without the harder stuff, we will add that in later.

-Blake Edwards / Dark Element
"""
import numpy as np
np.random.seed(420)#For ease of testing

from keras.datasets import mnist
#from keras.layers import Input, Dense, Activation, Flatten, Reshape, Merge
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K

"""
First, we load and prepare our data
"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

"""
Convert to float32
"""
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

"""
Feature Scale
"""
X_train /= 255
X_test /= 255

"""
Create One-hot label matrices
"""
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

"""
Now that data is prepared, we start our model.
Since this isn't a normal Sequential Keras model, we have to define our own and then wrap it in Keras' Model API at the end.
We do this by defining a symbolic graph
"""
"""
Define inputs, 
    X as size 784, Since it is of type Input it automatically has a None axis at the front
    T as size 3x3 since this is our transformation matrix
"""

input_dims = [28,28,3,]
capsule_n = 10
inputs = Input(shape=input_dims)

"""
First off, each capsule's hidden units are independent from one another.
This means we don't share the recognition units, so if we want to have n capsules easily,
    we have to make a MetaDense layer, so that we can say we want n groups of 20 hidden units / dense layers.
So our first model is only going to be testing this.
"""

"""
Get the dimensions of our recognition hidden flattened dims via the # of capsules, and the flattened input dimensions.
We also set the image dims in the same fashion, but without flattening our input_dims
"""
recognition_hidden_flattened_dims = [capsule_n, np.prod(input_dims)]
recognition_hidden_image_dims = [capsule_n]
recognition_hidden_image_dims.extend(input_dims)

"""
Flatten input dimensions and store in flattened_inputs
"""
flattened_inputs = Flatten()(inputs)

"""
Initialize output of our recognition hidden layer as a list
"""
recognition_hidden_flattened_outputs = []

"""
Then, thanks to Keras's functional model API, we can actually just loop through
    for each capsule, and apply an independent Dense Layer with activation function of our choice to the flattened inputs.
NOTE: For now having the output be the same size as the input, but this needs to be smaller or else we defeat the purpose of an autoencoder.
"""
for capsule_i in range(capsule_n):
    recognition_hidden_flattened_outputs.append(Dense(np.prod(input_dims))(flattened_inputs))

"""
We then pack our list into a tensor that we can manipulate further down our graph.
"""
recognition_hidden_flattened_outputs = merge([recognition_hidden_flattened_output for recognition_hidden_flattened_output in recognition_hidden_flattened_outputs], mode='concat')
#recognition_hidden_flattened_outputs = K.pack(recognition_hidden_flattened_outputs)

"""
We also reshape back into an image, now that we have put it through the Dense Layer
"""
recognition_hidden_image_outputs = Reshape(recognition_hidden_image_dims)(recognition_hidden_flattened_outputs)

"""
For this small example, assign what should be a result later in our model to this now since we aren't including the entire model yet.
"""
atomic_capsule_outputs = recognition_hidden_image_outputs

"""
Assign our final output value(s) to the sum of our atomic_capsule_outputs values over the capsule axis, axis 0
"""
"""
FIGURING IT OUT HERE
"""
composite_capsule_output = K.sum(atomic_capsule_outputs, axis=0)
#composite_capsule_output = merge([atomic_capsule_output for atomic_capsule_output in atomic_capsule_outputs], mode='sum')
#composite_capsule_output = merge(atomic_capsule_outputs, mode='sum')

"""
With our inputs and outputs, create a Keras Model.
"""
model = Model(input=inputs, output=composite_capsule_output)
"""
Initialize a session and get our outputs with some random inputs of shape (batch, input_dims[0], input_dims[1])
sess = K.get_session()
sample_inputs = np.ones((1,input_dims[0], input_dims[1], input_dims[2]))
print sample_inputs
a = sess.run(composite_capsule_output, feed_dict={inputs: sample_inputs})
print a
print a.shape
"""

