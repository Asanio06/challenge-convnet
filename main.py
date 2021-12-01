import random
import numpy as np
from sympy import reduced_totient
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


def load_training_dataset(dataset_location='./dataset/',
                          return_format='numpy',
                          image_size=(100, 100),
                          batch_size=32,
                          shuffle=True):
    '''
    Charge et retourne un dataset à partir d’un dossier contenant
    des images où chaque classe est dans un sous-dossier.
    Le dataset est peut être renvoyé comme deux tableaux NumPy, sous
    la forme d’un couple (features, label) ; ou comme un Dataset
    TensorFlow (déjà découpé en batch).
    # Arguments
        dataset_location: chemin vers le dossier contenant les images
            réparties dans des sous-dossiers représentants les
            classes.
        return_format: soit `numpy` (le retour sera un couple de
            tableaux NumPy (features, label)), soit `tf` (le
            retour sera un Dataset TensorFlow).
        image_size: la taille dans laquelle les images seront
            redimensionnées après avoir été chargée du disque.
        batch_size: la taille d’un batch, cette valeur n’est utilisée
            que si `return_format` est égale à `tf`.
        shuffle: indique s’il faut mélanger les données. Si défini à
            `False` les données seront renvoyées toujours dans le
            même ordre.
    # Retourne
        Un couple de tableaux NumPy (features, label) si
        `return_format` vaut `numpy`.
        Un Dataset TensorFlow si `return_format` vaut `tf`.
    '''
    ds = image_dataset_from_directory(
        dataset_location,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle if return_format == 'tf' else False,
        image_size=image_size,
        color_mode='rgb',
        interpolation='bilinear'
    )
    if return_format == 'tf':
        return ds
    elif return_format == 'numpy':
        X = np.concatenate([images.numpy() for images, labels in ds])
        y = np.concatenate([labels.numpy() for images, labels in ds])
        if shuffle:
            idx = list(range(len(X)))
            random.shuffle(idx)
            X = X[idx]
            y = y[idx]
        return (X, y)
    else:
        raise ValueError(
            'The `return_format` argument should be either `numpy` (NumPy arrays) or `tf` (TensorFlow dataset).')


if __name__ == "__main__":
    image_size = (100, 100)
    (X, Y) = load_training_dataset(image_size=image_size)
    ds = load_training_dataset(image_size=image_size, return_format='tf')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0

    num_classes = 3
    input_shape = (100, 100, 3)

    model = keras.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='valid'),

            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='valid'),

            layers.Flatten(),
            layers.Dropout(0.2),

            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    opt = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, batch_size=100, epochs=50, validation_data=(X_test, y_test))

    prediction_probas = model.evaluate(X_test, y_test)

    print(prediction_probas)
