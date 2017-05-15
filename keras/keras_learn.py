import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, np
from keras.optimizers import RMSprop
from keras.utils import np_utils
from load_data import load_retinopathy_data
import matplotlib.pyplot as plt


def load_mnist_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('size:', X_train.shape)
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    return (X_train, y_train), (X_test, y_test), num_pixels, num_classes


def main():
    print('Loading dataset')
    (X_train, y_train), (X_test, y_test), shape = load_retinopathy_data()
    print('Dinished loading dataset')
    if len(X_train) == 0:
        print('Empty training set')
        return
    print('Train set size:', len(X_train))

    # define baseline model
    def baseline_model():
        num_pixels = np.prod(shape)
        model = Sequential()
        model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))  # num_pixels
        model.add(Dropout(0.5))
        model.add(Dense(80,kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(40,kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
        # Compile model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # build the model
    model = baseline_model()
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=100, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


if __name__ == '__main__':
    main()
