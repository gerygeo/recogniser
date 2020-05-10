import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import models, layers


class CNNModel:

    def __init__(self, rows, cols, num_classes):
        self.img_rows = rows
        self.img_cols = cols
        self.num_classes = num_classes

    def process_data(self, trainfile, testfile):
        df_train = pd.read_csv(trainfile)
        df_train.reset_index()
        X_test = pd.read_csv(testfile)

        Y_train = tf.keras.utils.to_categorical(df_train.label, self.num_classes)

        df_train.drop('label', axis=1, inplace=True)
        X_train = df_train.to_numpy().reshape(df_train.shape[0], self.img_rows, self.img_cols, 1)
        X_test = X_test.to_numpy().reshape(X_test.shape[0], self.img_rows, self.img_cols, 1)

        # Normalize the data
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        return X_train, Y_train, X_test

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3)
                                , activation='relu'
                                , input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    def predict(self, model, X_train, Y_train, X_test):
        model.fit(X_train, Y_train, batch_size=128, epochs=12,
                  validation_split=0.2)
        preds = model.predict(X_test)
        labels = np.argmax(preds, axis=1)
        df = pd.DataFrame(range(1, preds.shape[0] + 1), columns=['ImageId'])
        df['Label'] = labels

        df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    cnn = CNNModel(28, 28, 10)
    # Pre-process training and test data
    X_train, Y_train, X_test = cnn.process_data('train.csv', 'test.csv')
    # Create and compile cnn model
    model = cnn.create_model()
    # Fit the training data to the model and save the predictions
    cnn.predict(model, X_train, Y_train, X_test)
