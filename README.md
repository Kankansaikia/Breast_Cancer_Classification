# Breast Cancer Classification using Neural Network

This project utilizes a neural network to classify breast cancer tumors into malignant or benign categories using the Wisconsin Breast Cancer dataset.

## Overview

The project includes the following steps:

1. **Loading and Preparing Data:**
   - The breast cancer dataset is loaded from `sklearn.datasets` and converted into a Pandas DataFrame.
   - Data exploration techniques such as checking for null values, data summary, and understanding class distributions are performed.

2. **Data Preprocessing:**
   - Features and target (`label`) are separated from the DataFrame.
   - Data is split into training and testing sets using `train_test_split`.
   - Standardization is applied to the data using `StandardScaler` from `sklearn.preprocessing`.

3. **Building and Training the Neural Network:**
   - A neural network model is constructed using TensorFlow/Keras.
   - The model architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with sigmoid activation for binary classification.
   - The model is compiled with `adam` optimizer and `sparse_categorical_crossentropy` loss function.
   - Training is performed on the training data with validation split and for a specified number of epochs.

4. **Model Evaluation:**
   - Model performance is evaluated using the test data, and accuracy metrics are calculated.
   - Loss and accuracy trends are visualized using Matplotlib.

5. **Prediction:**
   - A simple interface is provided to enter new data points (features of a tumor).
   - The input data is standardized and fed into the trained model for prediction.
   - Predictions are made and interpreted to classify tumors as malignant or benign.

## Files Included

- `breast_cancer_nn.ipynb`: Jupyter notebook containing the complete code.
- `breast_cancer.ipynb`: Jupyter Notebook containing the code using KNN algorithm 
- `README.md`: This file explaining the project.
- Any additional files (like data files or saved model files) if used.

## Libraries Used

- `numpy` for numerical computations.
- `pandas` for data manipulation and analysis.
- `matplotlib` for data visualization.
- `sklearn` for dataset loading, preprocessing, and evaluation.
- `tensorflow` and `keras` for building and training the neural network.

## Usage

To run the project:
1. Ensure you have Python installed along with the necessary libraries (`numpy`, `pandas`, `matplotlib`, `sklearn`, `tensorflow`, `keras`).
2. Execute the code in `breast_cancer_nn.ipynb` in a Jupyter notebook environment or any Python IDE.

## Conclusion

This project demonstrates the application of a neural network for breast cancer classification, achieving high accuracy in predicting tumor malignancy based on tumor characteristics.
