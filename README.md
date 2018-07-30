## Digit Recognizer

Built for python3. Uses TensorFlow Library. Installation instructions can be found at [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/).

Trains a model to recognize hand-written digits. Uses training data from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

Included are two implementations. The first uses a simple Neural Network with a single output layer. It achieves around 90% accuracy. 
The second implementation uses a Convolutional Neural Network which adds two convolution and pool layers. This implementation achieves over 98% accuracy. 

### Simple Neural Net Implementation
```
python3 simple-neural-net-digit-recognizer.py
```

### Convolutional Neural Net Implementation
```
python3 convolutional-neural-net-digit-recognizer.py
```


Based on Jerry Kurata's Pluralsight tutorial "TensorFlow: Getting Started"