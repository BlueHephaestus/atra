"""
Functions exclusively for the purpose of dealing with affine transformations in ATRA,
    from generating training data, to applying transformations, to anything else.

I would use other libraries (probably this one https://github.com/aleju/imgaug) if I didn't need to create my own transformation matrices / mechanisms for applying the transformations to our data.
    Fortunately, I don't need to include all the code for applying transformations here, as we can do that with OpenCV's WarpAffine() function.

This makes things much easier, and shortens the code in this file a great deal.

-Blake Edwards / Dark Element
"""

def generate_2d_transformed_data(data, sigma=0.1, transformation_n=9, transformation_matrices=None)
    """
    Arguments:
        data: 
            A np array of shape (N, H, W, C),
                N: Number of samples in the data
                H: Height of each sample
                W: Width of each sample
                C: Channels in each sample

            This is the data to have affine transformations applied, transformed, and returned with.
            Be careful, if you pass in 60,000 images, with transformation_n=9, you will end up with 10*60,000 = 600,000 result images.
            This can also be thought of as the X argument in a normal dataset.

        sigma: 
            Defaults to 0.1, this is a parameter to control how large the affine transformations will be when randomly generating.
                ->In Sida Wang's paper he makes use of a parameter of 0.05 or 0.1.
            Our formula where this is used is as follows:

                [1 0 0]           [r r r]
            T = [0 1 0] + sigma * [r r r]
                [0 0 1]           [0 0 0]

            Where:
                T is each transformation matrix,
                and each r is drawn from the standard normal distribution.

        transformation_n:
            Defaults to 9, so that if each sample in the data has 9 transformed versions of it created, the data size will increase by 10x as a result.
            Used with sigma, this is the number of randomly generated transformation matrices to create, in the random generation process.

        transformation_matrices: Optional - A np array of shape (N, 3, 3),
            N: Number of transformation matrices
            and 3x3 for each 2d transformation matrix
            
            If not supplied, will default to using the `sigma` argument and `transformation_n` arguments.
            If supplied, will take precedence over sigma and transformation_n arguments.
            Only supply this if you want to use your own specific transformation matrices.

    Returns:
        data:
            The original data argument.

        transformed_data:
            A np array of shape (N*transformation_n, H, W, C),
                With all dimensions the same as data, except for the first dimension, which is now N*transformation_n
                    due to the amount of transformed versions of the original input which are created.
                The samples here retain the order of the original data argument, s.t.:
                    if data[0] is a picture of the number 7 (i.e. with label 7), transformed_data[0:transformation_n+1] will also be labelled as 7s.
            This does not include the original data, that can be found in the first return value.
            For an example of 4 transformation matrices, the resulting data will be like the following example:
                [transformation1(data[0]), transformation2(data[0]), transformation3(data[0]), transformation4(data[0]), transformation1(data[1]), transformation2(data[1]), ...]
            So that the data retains it's original order, and the transformations repeat every transformation_n times.

        transformation_matrices:
            A np array of shape (transformation_n, 3, 3),
                containing the randomly generated transformation matrices if this function was called with sigma and transformation_n,
                or containing the original transformation_matrices if this function was called with those supplied.
    """
    if not transformation_matrices:
        """
        Randomly generate our transformation matrices using the method discussed above.
        """
        """
        First, we create our (transformation_n, 3, 3) matrices drawn from the standard normal
            by using np.random.randn and then inserting it into the correct location in a zeroed array.
        """
        transformation_matrices = np.zeros((transformation_n, 3, 3))
        transformation_matrices[:, :2, :] = np.random.randn(transformation_n, 2, 3)
        
        """
        Then multiply this by sigma and add all of it to an identity matrix of the same shape.
            We get our identity matrix by making a 3x3 one, then repeating this with python's use of the * operator.
        """
        transformation_matrices *= sigma
        transformation_matrices += np.array([np.eye(3)]*transformation_n) 

    """
    Regardless of option chosen, we now have transformation matrices.
    Now, we get the length of them to make sure we have the right number regardless of option 
    """
    transformation_n = len(transformation_matrices)

    """
    We then get the dimensions by replacing the first dimension of our data array with transformation_n,
        which we do via python's list's extend method on all dimensions but the first in our data array
    """
    transformed_data_dims = [transformation_n]
    transformed_data_dims.extend(data.shape[1:])

    """
    Using this, we generate a zeroed np array of our transformed data, since it is going to very likely take up a lot of memory.
    """
    transformed_data = np.zeros(transformed_data_dims)

    """
    Then, with all this ready, we loop through each sample in our original data and place it in the transformed data
        For any index in the original data array i,
        The transformed index j is computed as follows:
            j = i * transformation_n + transformation_i
        Which I have taken the liberty to prove, in the file transformed_data_index_proof.txt
    """


        

def generate_transformed_references(labels, transformation_matrices)
    """
    Arguments:
        labels:
            A np array of same size as data argument to generate_2d_transformed_data(), 
                so that the shape is (N, ...)
                with the remaining dimensions irrelevant.
            This contains the labels for the original data, and will be used to generate labels for the transformed data.

        transformation_matrices:
            A np array of the same type returned by generate_2d_transformed_data(),
                containing our transformation matrices used to compute our transformed data.
            Be careful when passing in different data than that returned by generate_2d_transformed_data().

    Returns:
        transformed_labels:
            A np array of shape (N*transformation_n, ...)
                With each label corresponding to transformed_data, instead of the original data.
                Assumes that generate_2d_transformed_data(), or one of our functions for generating transformed data,
                    was used to generate the transformed data. 
                If 4 matrices were supplied, result will look like:
                    [label1, label1, label1, label1, label2, label2, ...]

        transformation_matrices:
            A np array of shape (N*transformation_n, ...)
                With each transformation matrix corresponding to the matrix used to get the transformed data sample.
                If 4 matrices were supplied, result will look like:
                    [transformation1, transformation2, transformation3, transformation4, transformation1, transformation2, ...]

    Overrall, this function returns the modified labels and transformation matrices for easy reference with our transformed_data during training.
    """
