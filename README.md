# TensorFlow Keras MNIST Project

This is a boilerplate project for Image Classification using TensorFlow and Keras.

## Setup

> [!IMPORTANT]
> **Python Version**: This project requires Python 3.11 or older (compatible with TensorFlow).
> If your default python is newer (e.g., 3.14), use the Windows Launcher `py` to specify the version.

1. Create a virtual environment with Python 3.11:
   ```bash
   py -3.11 -m venv venv
   ```

2. Activate the environment:
   ```bash
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the model, run:
```bash
python src/train.py
```
This will:
- Load the MNIST dataset.
- Train the CNN model defined in `src/model.py`.
- Save the trained model to `model.h5`.
- Display a training history plot.

### Evaluation
To evaluate the trained model, run:
```bash
python src/evaluate.py
```
This will load `model.h5` and check its accuracy on the test set.

## Project Structure
- `src/model.py`: Model definition.
- `src/train.py`: Training script.
- `src/evaluate.py`: Evaluation script.
- `src/utils.py`: Visualization helpers.
