# hw5

All the tasks are to be completed using the keras Sequential interface. You can but are not required to use the scikit-learn wrappers included in keras. We recommend to run all but Task 1 on the habanero cluster which provides GPU support (though you could also run task 2 on a CPU). Running on your own machine without a GPU will take significantly longer. We have limited access to GPU resources on the cluster, so make sure to start working on the homework well in advance. Feel free to use any other compute resources at your disposal.

You can find instructions on accessing the cluster below.
- Task 1: 
    Run a multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the iris dataset using the keras Sequential interface. 

- Task 2: 
    Train a multilayer perceptron on the MNIST dataset. Compare a “vanilla” model with a model Qusing drop-out. Visualize the learning curves.

- Task 3:
    Train a convolutional neural network on the SVHN dataset in format 2 (single digit classification)
    
- Task 4:
    Load the weights of a pre-trained convolutional neural network, for example AlexNet or VGG, and use it as feature extraction method to train a linear model or MLP on the pets dataset. The pets dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/pets/ (37 class classification task).
