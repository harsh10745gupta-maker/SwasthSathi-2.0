# SwasthSathi 2.0

Advanced college mini-project: AI-assisted symptom screening using a Random Forest model.

## Dataset
Uses the Kaggle Disease Prediction Using Machine Learning dataset (`kaushil268/disease-prediction-using-machine-learning`). Place `Training.csv` and `Testing.csv` inside `dataset/`.

## Setup
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `python train_model.py`
5. Run: `python app.py`
6. Open `http://127.0.0.1:5000`

## Demo login
Email can be any email. OTP is `123456` by default.

Admin login: `/admin/login` and password `admin123` by default. Change these demo values for any real deployment.

## ML
The training script compares Decision Tree, Random Forest and Multinomial Naive Bayes on the supplied Testing.csv. Random Forest is used by the web application because it provides class probabilities and feature importance.

## Important
This is an educational prototype, not a medical diagnostic tool. Model scores are not clinical probabilities and should not be used to make treatment decisions.
