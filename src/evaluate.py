import tensorflow as tf
import os

def evaluate():
    if not os.path.exists('model.h5'):
        print("Model file not found. Please run train.py first.")
        return

    # Load the model
    model = tf.keras.models.load_model('model.h5')

    # Load dataset
    mnist = tf.keras.datasets.mnist
    (_, _), (test_images, test_labels) = mnist.load_data()

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images / 255.0

    # Evaluate
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Restored model, accuracy: {100 * acc:5.2f}%")

if __name__ == "__main__":
    evaluate()
