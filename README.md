

# Credit Risk ML Project

## 📖 Project Description
This project focuses on **predicting credit risk** of clients using **machine learning** techniques.  
A **Random Forest Classifier** is used, along with data preprocessing and feature engineering.

The project allows you to:
- Train a model on historical credit data  
- Save the trained model and feature names  
- Predict the risk of a new client and get recommendations for managing them  

---

##  Project Structure


```markdown
credit-risk-ml/
├── src/
│   ├── train.py        # Script to train the model
│   └── predict.py      # Script to predict risk for new clients
├── data/
│   └── german_credit_data.csv  # Dataset with historical credit data
├── model/             # Folder to save trained model and features (auto-created)
├── README.md
└── requirements.txt   # List of project dependencies

````

---

##  Installation & Dependencies

Recommended: **Python 3.10+**

Install dependencies:

```bash
pip install -r requirements.txt
````

**Example `requirements.txt` content:**

```
pandas
numpy
scikit-learn
joblib
```

---

##  How to Run

### 1. Train the model

```bash
python3 src/train.py
```

* Automatically loads dataset from `data/german_credit_data.csv`
* Splits data into train/test sets
* Trains a Random Forest model
* Prints evaluation metrics (ROC AUC, classification report)
* Saves model and feature names in the `model/` folder

### 2. Predict risk for a new client

```bash
python3 src/predict.py
```

* Uses the saved model
* Outputs:

  * Risk (High / Low)
  * Probability of default
  * Recommended recovery strategy

---

##  Notes

* If the `model/` folder does not exist, it will be created automatically when running `train.py`
* Dataset must be located at `data/german_credit_data.csv`
* Synthetic data can be used for testing if the real dataset is unavailable

---

## 📊 Example Metrics

Example output after training:

```
ROC AUC Score: 0.82
Classification Report:
              precision    recall  f1-score   support

       0           0.85      0.80      0.82        20
       1           0.78      0.83      0.80        20
```

---

## Author

Ayan Malik — student, Machine Learning and Data Analytics project

