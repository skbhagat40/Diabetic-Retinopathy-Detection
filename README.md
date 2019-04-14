# Diabetic-Retinopathy-Detection
A Convolutional Neural Network to Grade Diabetic Retinopathy Level Based on Retina Scan
This Project uses Keras a Deep Learning Library Based on Tensorflow.
The Model is Convolutional Neural Network with  Input being the Retina Scan and Output being the Diabetic Retinopathy grading with following labels
level 0 - NoDR
level 1 - Moderate DR
level 2 - Non Poleferative DR
level 3 - Poliferative DR
level 4 - Severe DR
Preprocessing Step -

All the pixel values of input images are mean normalized and scaled with the standard deviation.
The RGB Image is converted to GrayScale.
Image enhancement is done using histogram Equalization.

The Neural Network-

The Neural network consists of -
Convolutional Layers - These layers act like a filter and look for various features in the image.
BatchNormalization Layer - These layers are used to noramlize the gradient and help counter the problem of vanishing gradient.
Dropout Layer - This layer randomly drops several connections which helps in reducing overfitting.
Dense Layer - These are fully conected layers. The output of convolutional layer is flattened and fed into Dense Layer.

Activation Functions -

1. ReLU - Rectified Linear Unit is used in Convolutional Layers.
2. Softmax is used in final Output layer which classifes the input in five classes.

Optimizer-
SGD - Stochiastic Gradient Descent Optimizer is Used.

The architecture used is Residual Network which uses skip connections to deal with the problem of skip connections.
