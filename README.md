# Customer Churn Prediction Web App

This repository contains a machine learning project for predicting customer churn using an Artificial Neural Network (ANN). The project includes data preprocessing, model training, and a Streamlit web application for interactive churn prediction.

---

## Project Overview

Customer churn prediction is a critical task for businesses to identify customers who are likely to leave. This project uses a dataset of bank customers to build an ANN model that predicts whether a customer will churn based on various features such as credit score, geography, gender, age, balance, and more.

---

## Features

- Data preprocessing including label encoding and one-hot encoding of categorical variables.
- Feature scaling using StandardScaler.
- ANN model built with TensorFlow/Keras.
- Model training with early stopping and TensorBoard integration.
- Saving and loading of trained model, encoders, and scalers using pickle.
- Streamlit-based interactive web app for real-time churn prediction.

---

## Dataset

The dataset used is `Churn_Modelling.csv` with the following key features:

| Feature          | Description                            |
|------------------|------------------------------------|
| CreditScore      | Customer's credit score              |
| Geography       | Customer's country (France, Spain, Germany) |
| Gender          | Customer gender                     |
| Age             | Customer age                       |
| Tenure          | Number of years with the bank       |
| Balance         | Account balance                    |
| NumOfProducts   | Number of bank products used        |
| HasCrCard       | Whether the customer has a credit card (0 or 1) |
| IsActiveMember  | Whether the customer is an active member (0 or 1) |
| EstimatedSalary | Estimated salary                   |
| Exited          | Target variable: 1 if customer churned, 0 otherwise |

---

## Installation

1. Clone the repository:

   ```
   git clone 
   cd 
   ```

2. Create and activate a Python environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

   Required packages include:
   - pandas
   - scikit-learn
   - tensorflow
   - streamlit
   - numpy
   - pickle-mixin (if needed)

---

## Usage

### Training the Model

- The Jupyter notebook contains the full code to preprocess the data, encode categorical variables, scale features, build and train the ANN model.
- The model, encoders, and scaler are saved as `.h5` and `.pkl` files respectively for later use.

### Running the Streamlit App

- Run the Streamlit app locally with:

  ```
  streamlit run app.py
  ```

- The app allows users to input customer features interactively and outputs the predicted churn probability along with a simple interpretation.

---

## File Structure

| File              | Description                                      |
|-------------------|------------------------------------------------|
| `Churn_Modelling.csv` | Original dataset                              |
| `notebook.ipynb`  | Jupyter notebook with data preprocessing and model training code |
| `model.h5`        | Trained ANN model saved in HDF5 format          |
| `label_encoder_gender.pkl` | Pickled label encoder for Gender          |
| `onehot_encoder_geo.pkl`   | Pickled one-hot encoder for Geography     |
| `scaler.pkl`      | Pickled StandardScaler for feature scaling      |
| `app.py`          | Streamlit web application code                    |
| `requirements.txt`| List of required Python packages                  |

---

## Model Architecture

- Input layer: 12 features after encoding and scaling
- Hidden layers:
  - Dense layer with 64 neurons, ReLU activation
  - Dense layer with 32 neurons, ReLU activation
- Output layer:
  - Dense layer with 1 neuron, sigmoid activation (binary classification)

---

## Results

- The model achieves approximately 86% accuracy on the validation set.
- Early stopping is used to prevent overfitting.
- TensorBoard is integrated for training visualization.

---

## Acknowledgments

- Dataset source: [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)
- TensorFlow and Keras for deep learning framework
- Streamlit for building the interactive web app

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or suggestions, please open an issue or contact the repository owner.

---
```
