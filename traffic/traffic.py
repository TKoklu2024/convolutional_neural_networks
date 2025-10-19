import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Returns tuple `(images, labels)`. `images` will be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` will
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    images_label = (images, labels)
    
    """Check if the 'data_dir' value is a valid string object"""
    if isinstance(data_dir, str):
        directory = os.path.join(os.getcwd(), data_dir)
    else:
        raise TypeError("Directory, data_dir, must be a string.")
    
    """Get all images in each category and store them as numpy.ndarray"""
    for each in range(NUM_CATEGORIES):
        for img in os.listdir(os.path.join(directory, str(each))):  
            current_dir = os.path.join(directory, str(each))
            image_dir = os.path.join(current_dir, str(img))

            """Read an image with cv2 and resize it"""
            original_image = cv2.imread(image_dir)
            resized_image = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))
            # print(each, img, type(resized_image), len(resized_image), np.ndim(resized_image))
            images.append(resized_image)
            labels.append(int(each))
    return images_label


def get_model():
    """
    Returns a compiled convolutional neural network model. The
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer will have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network (CNN)
    model = tf.keras.models.Sequential([

        # Convolutional layer, learn 32 filters using a 3X3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 3X3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 43 different traffic signs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
    # raise NotImplementedError


if __name__ == "__main__":
    main()
