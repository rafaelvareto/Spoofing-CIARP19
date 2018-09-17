from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD


class DeepLearning:
    @staticmethod
    def build_LeNet(width, height, depth, nclasses, weightsPath=None):
        # initialize the model
        model = Sequential()
        print((width, height, depth))

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=(5, 5), padding="same", activation='relu', input_shape=(width, height, depth)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=(5, 5), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        # softmax classifier
        model.add(Dense(nclasses, activation='softmax'))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # initialize the optimizer and model
        opt = SGD(lr=0.01)  
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    @staticmethod
    def build_network_1(width, height, depth, nclasses, weightsPath=None):
        pass

    @staticmethod
    def build_network_2(width, height, depth, nclasses, weightsPath=None):
        pass
