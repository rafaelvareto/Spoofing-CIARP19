from keras import preprocessing
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


class DeepLearning:
    @staticmethod
    def build_LeNet(width, height, depth, nclasses, weightsPath=None):
        # initialize the model
        model = Sequential()
        input_shape = Input(shape=[width,height,depth])
        print([width,height,depth])

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), border_mode='same', name='conv1')(input_shape))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), border_mode='same', name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # initialize the optimizer and model
        opt = SGD(lr=0.01)  
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model
