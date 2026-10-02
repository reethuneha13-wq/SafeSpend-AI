# SafeSpend AI

## Intelligent Financial Transaction Risk Monitoring Platform

SafeSpend AI is a machine-learning-based transaction risk monitoring application that identifies unusual transaction patterns and assigns a risk score from 0 to 100.

A high risk score does not confirm that a transaction is fraudulent.

## Features

- CSV transaction upload
- Data validation
- Data cleaning
- Z-Score anomaly detection
- IQR anomaly detection
- Isolation Forest
- Local Outlier Factor (LOF)
- Time-based anomaly detection
- Risk score from 0 to 100
- Risk levels
- Transaction explanations
- Interactive visualizations
- High-risk transaction detection
- CSV results download
- Streamlit dashboard

## Machine Learning Methods

### Z-Score
Identifies transaction amounts that are unusually far from the average.

### IQR
Identifies transactions outside the normal spending range.

### Isolation Forest
An unsupervised machine-learning algorithm used to identify unusual observations.

### Local Outlier Factor
Detects observations that differ from nearby data points.

### Time-Based Detection
Transactions occurring before 6 AM are treated as an additional unusual-pattern signal.

## Risk Levels

| Score | Risk Level |
|---|---|
| 0-30 | Low |
| 31-60 | Medium |
| 61-80 | High |
| 81-100 | Very High |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Project Structure

SafeSpend-AI/
    app.py
    README.md
    requirements.txt
    .gitignore
    models/
        isolation_forest_model.pkl
        lof_model.pkl
    data/
        sample_transactions.csv
    images/

## How to Run

Install the required packages:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

Then upload a CSV containing:

    transaction_id
    amount
    category
    hour
    location

## Important Disclaimer

SafeSpend AI is an educational and project-level transaction risk monitoring system.

It identifies unusual patterns using statistical and machine-learning methods. It does not guarantee that a transaction is fraudulent.

Users should review flagged transactions using appropriate financial-security procedures.

Do not upload passwords, PINs, CVVs, OTPs, or other authentication credentials.

## Future Improvements

- Real-time transaction monitoring
- More behavioral features
- Improved model calibration
- User authentication
- Database integration
- Secure API backend
- Email notifications
- Mobile application
- Model retraining pipeline
- Production-grade security and privacy controls

## Project Status

Prototype / Educational Project

Built using Python, Machine Learning and Streamlit.
