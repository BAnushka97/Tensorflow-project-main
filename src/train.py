import tensorflow as tf
from model import create_model
from utils import plot_history
import os

def train():
    # Load and prepare the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images / 255.0

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images / 255.0

    # Create and train the model
    model = create_model()
    
    history = model.fit(train_images, train_labels, epochs=5, 
                        validation_data=(test_images, test_labels))

    # Save the model
    model.save('model.h5')
    print("Model saved to model.h5")

    # Plot history
    plot_history(history)

if __name__ == "__main__":
    train()
