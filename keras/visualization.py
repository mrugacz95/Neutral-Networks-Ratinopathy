from keras.models import model_from_json


def main():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    from keras.utils import plot_model
    plot_model(loaded_model, to_file='model.png')


if __name__ == '__main__':
    main()