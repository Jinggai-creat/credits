import tensorflow
import keras
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt
import os

# Where to save the figures (i.e. /content/images/Topic_4)
PROJECT_ROOT_DIR = "."
PATH = os.path.join(PROJECT_ROOT_DIR, "data")
os.makedirs(PATH, exist_ok=True)
DATA_PATH = "C:\workspace\Demo\credicard\creditcard.csv"


def save_data(data, data_name, tight_layout=True, fig_extension="csv", resolution=300):
    path = os.path.join(PATH, data_name + "." + fig_extension)
    data.to_csv(path, index=False)


def data_reader():

    # read data from file
    data = pd.read_csv(DATA_PATH)
    class_data = data['Class']
    print(class_data.unique())
    print(data.shape)
    print(data)
    return data


def generate_training_testing_data(data):
    # Select 80% as training set, 20% as testing set
    random_num = np.random.rand(len(data)) < 0.8
    x_train_data = data[random_num]
    print(x_train_data)

    x_test_data = data[~random_num]

    return x_train_data, x_test_data


if __name__ == '__main__':
    label_data = data_reader()
    x_train, x_test = generate_training_testing_data(label_data)

    print(x_train)

    y_train = x_train.pop('Class')
    y_test = x_test.pop('Class')
    # save_data(x_train,"x_train")

    # build the model
    model = keras.models.Sequential([
        keras.layers.Dense(30, name="l1"),
        keras.layers.Dense(4, activation='relu', name="l2"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax', name="l3")
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    model.fit(x_train.values, y_train.values, epochs=1)

    # test the model
    test_loss, test_acc = model.evaluate(x_test.values, y_test.values, verbose=2)
    print('\nTest accuracy:', test_acc)

    MODEL_PATH = PATH = os.path.join(PROJECT_ROOT_DIR, "exp4_model")
    print(MODEL_PATH)
    model.save(MODEL_PATH);

    # probability_model = keras.Sequential([model,
    #                                          keras.layers.Softmax()])
    #
    # # Use the model to predict the class of the test data
    # predictions = probability_model.predict(x_test.values)
    #
    # print(predictions)

