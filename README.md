#Description 

The majority of the explanation behind this method and project will be posted in a series on my blog [here](https://dark-element.com/), when I make it. For now, I am still figuring out the intricacies of the method, and the project will be divided into a Mark system. I wrote this in Keras, with tendency towards the Tensorflow backend.

##MK 1.1

Test model, makes use of the Keras functional model API to have variable number of capsules for given inputs, apply a dense layer (output of each = activation(weight * input + bias)) to the input for each capsule, get all the outputs in one tensor, and then sum those outputs together into a final output.

This model consists of only one composite capsule, however can have n atomic capsules inside of it. Obviously, complete atomic capsules are much more complex than this one currently is, since this one essentially treats each atomic capsule as being one dense layer with a constant output size throughout all capsules (missing gating, transformations, etc)

Trained with MNIST data for debugging, in the normal autoencoder fashion where it tries to recreate the input.





This README will be extended in the future, as this project is still in its very early stages.
