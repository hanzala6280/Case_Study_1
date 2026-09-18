# 🏥 Hospital Readmission Prediction

This project uses **Logistic Regression** to predict whether a patient will be readmitted to the hospital.

The notebook performs data preprocessing, categorical feature encoding, numerical feature scaling, model training, and evaluation using the **ROC-AUC score** and a **confusion matrix**.

## 📌 Project Overview

The objective is to build a binary classification model that predicts hospital readmission.

The target variable is:

* `yes` → Patient was readmitted
* `no` → Patient was not readmitted

The project uses **Logistic Regression with L2 regularization** as the classification algorithm.

## 📊 Dataset

The project uses a dataset named:

```text
hospital_readmissions.csv
```

The dataset contains patient and hospital-related information, including features such as:

* `age`
* `time_in_hospital`
* `n_lab_procedures`
* `n_procedures`
* `n_medications`
* `medical_specialty`
* `diag_1`
* `diag_2`
* `diag_3`
* `glucose_test`
* `A1Ctest`
* `change`
* `diabetes_med`
* `readmitted`

## 🔄 Machine Learning Workflow

```text
Hospital Readmission Dataset
            ↓
       Load Dataset
            ↓
     Data Exploration
            ↓
   Check Missing Values
            ↓
    Check Duplicates
            ↓
 Encode Target Variable
            ↓
 One-Hot Encode Categorical Data
            ↓
     Train-Test Split
            ↓
    Standardize Numerical Data
            ↓
   Logistic Regression (L2)
            ↓
     Model Prediction
            ↓
 ROC-AUC + ROC Curve
            ↓
     Confusion Matrix
```

## 🧹 Data Preprocessing

### 1. Target Encoding

The `readmitted` column is converted from categorical values into binary values:

```python
df['readmitted'] = df['readmitted'].apply(
    lambda x: 1 if x == 'yes' else 0
)
```

Therefore:

```text
yes → 1
no  → 0
```

### 2. Categorical Encoding

The following categorical columns are converted using **one-hot encoding**:

```python
categorical_cols = [
    'age',
    'medical_specialty',
    'diag_1',
    'diag_2',
    'diag_3',
    'glucose_test',
    'A1Ctest',
    'change',
    'diabetes_med'
]
```

`drop_first=True` is used to avoid redundant dummy variables.

### 3. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A random state of `42` is used, and `stratify=y` maintains the class distribution between training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 4. Feature Scaling

Numerical features are standardized using `StandardScaler`.

The scaler is fitted only on the training data and then applied to the test data.

```python
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

## 🤖 Machine Learning Model

The project uses **Logistic Regression** with **L2 regularization**.

```python
model = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    random_state=42,
    max_iter=1000
)
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

## 📈 Model Evaluation

The model is evaluated using two main approaches.

### ROC-AUC Score

The model generates probability predictions using:

```python
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

The **ROC-AUC score** is then calculated:

```python
roc_auc = roc_auc_score(
    y_test,
    y_pred_proba
)
```

A ROC curve is also plotted to visualize the relationship between:

* True Positive Rate
* False Positive Rate

### Confusion Matrix

The model's predicted classes are compared with the actual classes using a confusion matrix.

The confusion matrix helps identify:

* True Positives
* True Negatives
* False Positives
* False Negatives

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
Hospital-Readmission-Prediction/
│
├── Case_1.ipynb
├── hospital_readmissions.csv
└── README.md
```

## 🚀 How to Run

### 1. Install the required libraries

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### 2. Place the dataset in the appropriate location

The notebook currently loads the dataset from:

```text
/content/hospital_readmissions.csv
```

This path is suitable for Google Colab.

### 3. Open the notebook

Open:

```text
Case_1.ipynb
```

Run the cells sequentially.

## 🎯 Key Concepts Demonstrated

### Binary Classification

The model predicts one of two outcomes:

```text
0 → Not readmitted
1 → Readmitted
```

### Logistic Regression

A classification algorithm that estimates the probability of an observation belonging to a particular class.

### L2 Regularization

L2 regularization helps control model complexity by penalizing large model coefficients.

### One-Hot Encoding

Converts categorical variables into numerical binary features that can be used by the machine learning model.

### Standardization

Transforms numerical features so they are on a comparable scale.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

Provides a detailed view of correct and incorrect predictions for each class.

## ⚠️ Note

This project is intended for **educational and machine-learning practice purposes**. The notebook demonstrates the complete workflow from preprocessing to model evaluation, but additional validation and analysis would be required before using such a model in a real healthcare environment.

## 👨‍💻 Author

**Hanzala**

B.Tech - Artificial Intelligence & Machine Learning
