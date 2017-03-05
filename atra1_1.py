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
    X as size (None, 28, 28) Since it is of type Input it automatically has a None axis at the front,
        and this will become (capsule_n, None, 28, 28).
        This results in our format of Capsule_N x Batch Size x Height x Width x Channels.
        Of course, we can easily manipulate the input dimensions to have (784,) or (28,28) or (784, 1, 1) or (28, 28, 1), and so on.
"""

capsule_n = 2
input_dims = [28,28]
inputs = Input(shape=input_dims)

"""
First off, each capsule's hidden units are independent from one another.
This means we don't share the recognition units, so if we want to have n capsules easily,
    we have to make a MetaDense layer, so that we can say e.g. we want n groups of 20 hidden units, aka dense layers.
So our first model is only going to be testing this.
"""

"""
TEMPORARILY DEPRECATED
Get the dimensions of our recognition hidden flattened dims via the # of capsules, and the flattened input dimensions.
We also set the image dims in the same fashion, but without flattening our input_dims
"""
"""
recognition_hidden_flattened_dims = [capsule_n, np.prod(input_dims)]
recognition_hidden_image_dims = [capsule_n]
recognition_hidden_image_dims.extend(input_dims)
"""

"""
We don't have the capsule number included in our input dimensions, but it is an extra hyper parameter.
However, since it affects the output shape, namely, NxHxW in this case, we have to manipulate our dimensions with it in mind.
    So, we can't flatten normally. 
    We have a Lambda layer here to reshape according to our recognition_hidden_flattened_dims, instead.
When we put our inputs through the introductory recognition hidden layer in each of our capsules when defining our model, 
    we will be looping through each capsule and applying the same input to a different dense layer each time.
Because we are putting the inputs, of shape (None, 28, 28) (where None = Mini batch size), 
    through these dense layers, we want to flatten our inputs first. So we do that here.
"""
#flattened_inputs = Lambda(lambda inputs: K.reshape(inputs, [np.prod(input_dims)]))(inputs)
flattened_inputs = Flatten()(inputs)

"""
Initialize output of our recognition hidden layer as a list
"""
recognition_hidden_flattened_outputs = []

"""
Then, thanks to Keras's functional model API, we can actually just loop through
    for each capsule, and apply an independent Dense Layer with activation function of our choice to the flattened inputs.
NOTE: For now having the output be the same size as the input, but this needs to be smaller or else we defeat the purpose of an autoencoder.
NOTE2: This is because i'm not sure how we are expected to move the encoded input back into a larger input at the end yet
"""
for capsule_i in range(capsule_n):
    recognition_hidden_flattened_outputs.append(Dense(28*28, activation="relu")(flattened_inputs))

"""
We then merge all the elements in our list with mode concat so that we can manipulate them further down our graph.
"""
recognition_hidden_flattened_outputs = merge([recognition_hidden_flattened_output for recognition_hidden_flattened_output in recognition_hidden_flattened_outputs], mode='concat')

"""
We also reshape back into an image, now that we have put it through the Dense Layer
Now that we have put the data through each capsule's dense layer, we have 
    recognition_hidden_flattened_outputs as shape (None, capsule_n*#_of_dense_outputs).
        Where None is the mini batch size, not included in our calculations but definitely warrants mentioning.
We reshape this so that recognition_hidden_capsule_outputs becomes shape (None, capsule_n, #_of_dense_outputs),
    And we get the last dimension via K.int_shape(recognition_hidden_flattened_outputs)[-1]//capsule_n, 
        since according to my statement of the shape of recognition_hidden_flattened_outputs, this will return #_of_dense_outputs.
With this, we have divided the outputs according to capsule, so that the output tensor is of shape (mini batch, capsule, outputs #)
"""
#print K.int_shape(recognition_hidden_flattened_outputs)
#print K.int_shape(recognition_hidden_flattened_outputs)[-1]//capsule_n
recognition_hidden_capsule_outputs = Reshape([capsule_n, K.int_shape(recognition_hidden_flattened_outputs)[-1]//capsule_n])(recognition_hidden_flattened_outputs)

"""
TEMPORARY
If we're only outputting the same size as the original from our dense layers, 
    then we want to reshape these back into the image / matrix representations as they were originally given,
    so that we can properly apply transformations.
However, it should be noted that we could still apply transformations if they were reduced,
    however the encodings would have to be reshapeable to images, e.g. 28x28 -> 11x11, 160x290 -> 80x145
"""
recognition_hidden_image_outputs = Reshape([capsule_n, 28, 28])(recognition_hidden_capsule_outputs)

"""
For this small example, assign what should be a result later in our model to this now since we aren't including the entire model yet.
"""
atomic_capsule_outputs = recognition_hidden_image_outputs

"""
Note: The dimensions / axes are (Mini batches, Capsules, ...)
Assign our final output value(s) to the sum of our atomic_capsule_outputs values over the capsule axis, axis 1
    We can't just insert random theano / tensorflow operations into our Keras model. 
    Because of this, we have to put them inside a lambda layer, and then apply that lambda layer to our input to get our output.
    That is what we do here, summing over axis 1 for the given atomic_capsule_outputs tensor
"""
composite_capsule_output = Lambda(lambda atomic_capsule_outputs: K.sum(atomic_capsule_outputs, axis=1))(atomic_capsule_outputs)

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
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
results = model.fit(X_train[:100], X_train[:100], nb_epoch=80, batch_size=4, shuffle=True)
print results.history["loss"]
print results.history["acc"]
