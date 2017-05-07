import random
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from load_data import load_retinopathy_data

def main():
    (x_train, y_train), (X_test, y_test), num_pixels = load_retinopathy_data()  # mnist.load_data()
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # num_pixels = X_test.shape[1] * X_test.shape[2]
    print(X_test.shape)
    number_of_test = 10
    random_offset = random.randint(0, X_test.shape[0] - number_of_test)
    for i in range(random_offset, random_offset + number_of_test):
        input_data = X_test[i:i + 1]
        img = input_data[0].copy()
        img = np.reshape(img, (28, 28))
        plt.imshow(img, cmap='gray')
        # input = input.reshape(1, num_pixels).astype('float32')
        result = loaded_model.predict_proba(input_data, verbose=1)
        print(result, 'expected:', y_test[i])
        plt.show()


if __name__ == '__main__':
    main()
