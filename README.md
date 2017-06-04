# Description 

The majority of the explanation behind this method and project will be posted in a series on my blog [here](https://dark-element.com/), when I make it. For now, I am still figuring out the intricacies of the method, and the project will be divided into a Mark system. I wrote this in Keras, with tendency towards the Tensorflow backend.

## MK 1.1 - Initial Model

Test model, makes use of the Keras functional model API to have variable number of capsules for given inputs, apply a dense layer (output of each = activation(weight * input + bias)) to the input for each capsule, get all the outputs in one tensor, and then sum those outputs together into a final output.

This model consists of only one composite capsule, however can have n atomic capsules inside of it. Obviously, complete atomic capsules are much more complex than this one currently is, since this one essentially treats each atomic capsule as being one dense layer with a constant output size throughout all capsules (missing gating, transformations, etc)

It allows a variable number of capsules (as long as number is > 1, otherwise it's just a lonely smol neural network), variable mini batch size, and all the normal stuff you would be able to vary in a normal Keras Sequential model / feedforward neural network.

In each capsule, it puts the entire input through a dense relu layer, with 3x3 encoding output. It then puts the encoding through another dense layer, outputting the capsule's data's original input shape as output. The output from each capsule is merged together / concatenated, and then summed to get the final output. 

Trained with MNIST data for debugging, in the normal autoencoder fashion where it tries to recreate the input.

In summary, the features are:

  1. Variable Atomic Capsule Number > 1
  2. Variable Mini Batch Size
  3. One ReLU layer for Recognition Hidden Units
  4. One ReLU layer for Generation Hidden Units
  5. Sum over all capsule's output for final output

## MK 1.2 - Barebones main components

Building from MK 1.1's features, we added
  
  1. Dropout
  2. More modular adjustment of model parameters
  3. Display of sample autoencoder outputs
  4. Graphing of results
  5. Display of individual capsule outputs per sample
  6. Visual Entity Probabilities & Gating units

## MK 1.3 - Application of Transformations

Building from MK 1.2's features, this version will be all about implementing transformations.

Planning to add:
  1. Modular random generation of transformed training data for use with autoencoder
  2. Implementation of transformation into encoding of autoencoder for use with training

This README will be extended in the future, as this project is still in its very early stages.
