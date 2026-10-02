
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SafeSpend AI",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🛡️ SafeSpend AI")

st.subheader(
    "Intelligent Financial Risk & Anomaly Detection Platform"
)

st.write(
    "Upload your transaction CSV file to identify unusual "
    "spending patterns and calculate transaction risk."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("SafeSpend AI")

st.sidebar.write(
    "Analyze your financial transactions using "
    "statistical and machine learning methods."
)

st.sidebar.info(
    "SafeSpend AI identifies unusual patterns. "
    "An unusual transaction is not necessarily fraud."
)


# ==========================================
# CSV UPLOAD
# ==========================================

st.header("📂 Upload Transactions")

uploaded_file = st.file_uploader(
    "Upload your transaction CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("📁 Please upload a CSV file to begin analysis.")
    st.stop()


# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    try:

        # ==========================================
        # READ CSV
        # ==========================================

        df = pd.read_csv(uploaded_file)

        st.success("✅ CSV file uploaded successfully!")


        # ==========================================
        # REQUIRED COLUMNS
        # ==========================================

        required_columns = [
            "transaction_id",
            "amount",
            "category",
            "hour",
            "location"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]


        if len(missing_columns) > 0:

            st.error(
                "❌ Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.info(
                "Required columns: "
                + ", ".join(required_columns)
            )

            st.stop()


        st.success(
            "✅ All required columns are present."
        )


        # ==========================================
        # ORIGINAL DATA INFORMATION
        # ==========================================

        original_rows = len(df)

        original_duplicates = df.duplicated().sum()


        # ==========================================
        # DATA CLEANING
        # ==========================================

        st.subheader("🧹 Data Cleaning")


        # ------------------------------------------
        # 1. Remove duplicate rows
        # ------------------------------------------

        duplicate_rows = df.duplicated().sum()

        if duplicate_rows > 0:

            df = df.drop_duplicates()

            st.info(
                f"🧹 Removed {duplicate_rows} duplicate row(s)."
            )

        else:

            st.success(
                "✅ No duplicate rows needed to be removed."
            )


        # ------------------------------------------
        # 2. Convert amount to numeric
        # ------------------------------------------

        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce"
        )


        # Count invalid amounts
        invalid_amounts = df["amount"].isnull().sum()


        if invalid_amounts > 0:

            df = df.dropna(
                subset=["amount"]
            )

            st.warning(
                f"⚠️ Removed {invalid_amounts} row(s) "
                "with invalid or missing amounts."
            )

        else:

            st.success(
                "✅ All transaction amounts are valid."
            )


        # ------------------------------------------
        # 3. Remove negative amounts
        # ------------------------------------------

        negative_amounts = (
            df["amount"] < 0
        ).sum()

        if negative_amounts > 0:

            df = df[
                df["amount"] >= 0
            ]

            st.warning(
                f"⚠️ Removed {negative_amounts} "
                "negative transaction amount(s)."
            )

        else:

            st.success(
                "✅ No negative transaction amounts."
            )


        # ------------------------------------------
        # 4. Convert hour to numeric
        # ------------------------------------------

        df["hour"] = pd.to_numeric(
            df["hour"],
            errors="coerce"
        )


        invalid_hours = df["hour"].isnull().sum()


        if invalid_hours > 0:

            df = df.dropna(
                subset=["hour"]
            )

            st.warning(
                f"⚠️ Removed {invalid_hours} row(s) "
                "with missing or invalid hours."
            )

        else:

            st.success(
                "✅ All transaction hours are valid."
            )


        # ------------------------------------------
        # 5. Keep only valid hours
        # ------------------------------------------

        invalid_hour_range = (
            (df["hour"] < 0) |
            (df["hour"] > 23)
        ).sum()


        if invalid_hour_range > 0:

            df = df[
                (df["hour"] >= 0) &
                (df["hour"] <= 23)
            ]

            st.warning(
                f"⚠️ Removed {invalid_hour_range} "
                "row(s) with invalid hours."
            )

        else:

            st.success(
                "✅ All hours are between 0 and 23."
            )


        # ------------------------------------------
        # 6. Handle category
        # ------------------------------------------

        category_missing = df["category"].isnull().sum()

        if category_missing > 0:

            df["category"] = df["category"].fillna(
                "Unknown"
            )

            st.info(
                f"ℹ️ Filled {category_missing} "
                "missing category value(s) with 'Unknown'."
            )

        else:

            st.success(
                "✅ No missing category values."
            )


        # ------------------------------------------
        # 7. Handle location
        # ------------------------------------------

        location_missing = df["location"].isnull().sum()

        if location_missing > 0:

            df["location"] = df["location"].fillna(
                "Unknown"
            )

            st.info(
                f"ℹ️ Filled {location_missing} "
                "missing location value(s) with 'Unknown'."
            )

        else:

            st.success(
                "✅ No missing location values."
            )


        # ==========================================
        # RESET INDEX
        # ==========================================

        df = df.reset_index(drop=True)


        # ==========================================
        # CLEANING SUMMARY
        # ==========================================

        final_rows = len(df)

        rows_removed = original_rows - final_rows


        st.subheader("📊 Cleaning Summary")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Original Rows",
                original_rows
            )


        with col2:

            st.metric(
                "Clean Rows",
                final_rows
            )


        with col3:

            st.metric(
                "Rows Removed",
                rows_removed
            )


        # ==========================================
        # CHECK IF DATA STILL EXISTS
        # ==========================================

        if len(df) == 0:

            st.error(
                "❌ No valid transactions remain "
                "after cleaning."
            )

            st.stop()


        # ==========================================
        # CLEAN DATASET
        # ==========================================

        st.subheader(
            "✅ Clean Transaction Dataset"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # ---------------------------------------------------
        # DOWNLOAD SECTION 20 ML RESULTS
        # ---------------------------------------------------

        st.subheader("⬇️ Download ML Results")

        csv_data = df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Section 20 Results",
            data=csv_data,
            file_name="safespend_section20_results.csv",
            mime="text/csv"
        )


        # ==========================================
        # FINAL CHECK
        # ==========================================

        st.success(
            "🎉 Data cleaning completed successfully!"
        )

        st.info(
            "The cleaned dataset is ready for "
            "feature engineering and machine learning."
        )

        # ===================================================
        # SECTION 20 - ML ANOMALY DETECTION
        # ===================================================

        st.header("🤖 Section 20 — ML Anomaly Detection")

        st.write(
            "The cleaned transactions are now analyzed "
            "using the trained anomaly detection models."
        )

        # Load Isolation Forest model
        isolation_model = joblib.load(
            "SafeSpend-AI/models/isolation_forest_model.pkl"
        )

        # Load LOF model
        lof_model = joblib.load(
            "SafeSpend-AI/models/lof_model.pkl"
        )

        st.success(
            "✅ ML models loaded successfully!"
        )

        # Select features used by the models
        features = df[["amount", "hour"]]

        # Isolation Forest prediction
        isolation_prediction = isolation_model.predict(
            features
        )

        df["isolation_anomaly"] = (
            isolation_prediction == -1
        )

        # LOF prediction
        lof_prediction = lof_model.predict(
            features
        )

        df["lof_anomaly"] = (
            lof_prediction == -1
        )

        # Display results
        st.subheader(
            "🔍 Anomaly Detection Results"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # Count anomalies
        isolation_count = int(
            df["isolation_anomaly"].sum()
        )

        lof_count = int(
            df["lof_anomaly"].sum()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Isolation Forest Anomalies",
                isolation_count
            )

        with col2:
            st.metric(
                "LOF Anomalies",
                lof_count
            )

        st.info(
            "ℹ️ An anomaly means the transaction looks "
            "unusual compared with the patterns learned "
            "by the models. It does not prove fraud."
        )



    except Exception as e:

        st.error(
            "❌ An error occurred while processing "
            "the transaction data."
        )

        st.write(
            "Error:",
            str(e)
        )


else:

    st.info(
        "📁 Please upload a CSV file to begin analysis."
    )


# ===================================================
# SECTION 21 - RISK SCORE
# ===================================================

st.header("⚠️ Section 21 — Transaction Risk Score")

st.write(
    "A rule-based score from 0 to 100 is calculated "
    "using the detected unusual transaction patterns."
)

# Start risk score
df["risk_score"] = 0

# Z-score anomaly
if "z_anomaly" in df.columns:
    df.loc[df["z_anomaly"] == True, "risk_score"] += 30

# IQR anomaly
if "iqr_anomaly" in df.columns:
    df.loc[df["iqr_anomaly"] == True, "risk_score"] += 20

# Isolation Forest anomaly
df.loc[df["isolation_anomaly"] == True, "risk_score"] += 20

# LOF anomaly
df.loc[df["lof_anomaly"] == True, "risk_score"] += 20

# Unusual transaction time
df["time_anomaly"] = df["hour"] < 6
df.loc[df["time_anomaly"] == True, "risk_score"] += 10

# Keep score between 0 and 100
df["risk_score"] = df["risk_score"].clip(0, 100)


# Risk level function
def get_risk_level(score):

    if score <= 30:
        return "Low"

    elif score <= 60:
        return "Medium"

    elif score <= 80:
        return "High"

    else:
        return "Very High"


# Apply risk level
df["risk_level"] = df["risk_score"].apply(get_risk_level)


# Display results
st.subheader("📊 Risk Score Results")

st.dataframe(
    df[
        [
            "transaction_id",
            "amount",
            "hour",
            "isolation_anomaly",
            "lof_anomaly",
            "time_anomaly",
            "risk_score",
            "risk_level"
        ]
    ],
    width="stretch"
)


# Calculate statistics
average_risk = round(df["risk_score"].mean(), 2)

high_risk_count = int(
    (df["risk_score"] > 60).sum()
)


# Display metrics
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Risk Score",
        average_risk
    )

with col2:
    st.metric(
        "Transactions Above 60",
        high_risk_count
    )


st.info(
    "ℹ️ The risk score indicates unusual transaction "
    "patterns. It does not confirm that a transaction "
    "is fraudulent."
)


# ===================================================
# SECTION 22 - EXPLANATION ENGINE
# ===================================================

st.header("🔎 Section 22 — Transaction Explanation")

st.write(
    "This section explains the main reasons why a transaction "
    "was given its risk score."
)


# Function to generate explanation
def generate_explanation(row):

    reasons = []

    # Check unusual amount
    if "z_anomaly" in row.index and row["z_anomaly"] == True:
        reasons.append(
            "Transaction amount is unusually high"
        )

    # Check IQR anomaly
    if "iqr_anomaly" in row.index and row["iqr_anomaly"] == True:
        reasons.append(
            "Transaction is outside the normal spending range"
        )

    # Check Isolation Forest
    if row["isolation_anomaly"] == True:
        reasons.append(
            "Isolation Forest detected unusual behaviour"
        )

    # Check LOF
    if row["lof_anomaly"] == True:
        reasons.append(
            "Transaction differs from nearby spending patterns"
        )

    # Check unusual time
    if row["time_anomaly"] == True:
        reasons.append(
            "Transaction occurred at an unusual time"
        )

    # No unusual behaviour
    if len(reasons) == 0:
        return "No major unusual behaviour detected"

    return "; ".join(reasons)


# Generate explanations
df["explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# Display explanation results
st.subheader("🧠 Why was this transaction flagged?")

st.dataframe(
    df[
        [
            "transaction_id",
            "amount",
            "hour",
            "risk_score",
            "risk_level",
            "explanation"
        ]
    ],
    width="stretch"
)


# Select a transaction
st.subheader("🔍 Check Individual Transaction")

transaction_ids = df["transaction_id"].tolist()

selected_id = st.selectbox(
    "Select Transaction ID",
    transaction_ids
)


# Get selected transaction
selected_transaction = df[
    df["transaction_id"] == selected_id
].iloc[0]


# Display selected transaction
st.write("### Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Amount",
        f"₹{selected_transaction['amount']:.2f}"
    )

with col2:
    st.metric(
        "Risk Score",
        selected_transaction["risk_score"]
    )

with col3:
    st.metric(
        "Risk Level",
        selected_transaction["risk_level"]
    )


# Explanation box
st.write("### 🧠 Explanation")

with st.expander("Why was this transaction flagged?", expanded=True):

    st.write(
        selected_transaction["explanation"]
    )


# Show transaction information
st.write("### 📋 Transaction Information")

st.write(
    {
        "Transaction ID":
            selected_transaction["transaction_id"],

        "Category":
            selected_transaction["category"],

        "Location":
            selected_transaction["location"],

        "Hour":
            selected_transaction["hour"],

        "Risk Score":
            selected_transaction["risk_score"],

        "Risk Level":
            selected_transaction["risk_level"]
    }
)



# ===================================================
# SECTION 23 - VISUALIZATION DASHBOARD
# ===================================================

st.header("📊 Section 23 — Visualization Dashboard")

st.write(
    "Visual summaries of transaction risk and unusual patterns."
)


# ---------------------------------------------------
# 1. Risk Level Distribution
# ---------------------------------------------------

st.subheader("1️⃣ Risk Level Distribution")

risk_counts = df["risk_level"].value_counts()

st.bar_chart(
    risk_counts,
    width="stretch"
)


# ---------------------------------------------------
# 2. Anomaly Detection Comparison
# ---------------------------------------------------

st.subheader("2️⃣ Anomaly Detection Comparison")

isolation_count = int(
    df["isolation_anomaly"].sum()
)

lof_count = int(
    df["lof_anomaly"].sum()
)

time_count = int(
    df["time_anomaly"].sum()
)

anomaly_data = pd.DataFrame({
    "Detection Method": [
        "Isolation Forest",
        "LOF",
        "Unusual Time"
    ],
    "Number of Transactions": [
        isolation_count,
        lof_count,
        time_count
    ]
})

st.bar_chart(
    anomaly_data,
    x="Detection Method",
    y="Number of Transactions",
    width="stretch"
)


# ---------------------------------------------------
# 3. Transaction Amount Distribution
# ---------------------------------------------------

st.subheader("3️⃣ Transaction Amount Distribution")

amount_data = df[
    ["transaction_id", "amount"]
].set_index("transaction_id")

st.line_chart(
    amount_data,
    width="stretch"
)


# ---------------------------------------------------
# 4. Risk Score Distribution
# ---------------------------------------------------

st.subheader("4️⃣ Risk Score Distribution")

risk_data = df[
    ["transaction_id", "risk_score"]
].set_index("transaction_id")

st.line_chart(
    risk_data,
    width="stretch"
)


# ---------------------------------------------------
# 5. Top High-Risk Transactions
# ---------------------------------------------------

st.subheader("5️⃣ Top High-Risk Transactions")

top_risk = df.sort_values(
    "risk_score",
    ascending=False
).head(10)

st.dataframe(
    top_risk[
        [
            "transaction_id",
            "amount",
            "category",
            "hour",
            "location",
            "risk_score",
            "risk_level",
            "explanation"
        ]
    ],
    width="stretch"
)


# ---------------------------------------------------
# 6. Dashboard Summary
# ---------------------------------------------------

st.subheader("📌 Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        len(df)
    )

with col2:
    st.metric(
        "Isolation Anomalies",
        isolation_count
    )

with col3:
    st.metric(
        "LOF Anomalies",
        lof_count
    )

with col4:
    st.metric(
        "High Risk",
        int((df["risk_score"] > 60).sum())
    )



# ===================================================
# SECTION 24 - FINAL DASHBOARD DESIGN
# ===================================================

st.header("🏦 SafeSpend AI Dashboard")

st.write(
    "An intelligent transaction risk monitoring dashboard "
    "for identifying unusual spending patterns."
)

st.info(
    "⚠️ SafeSpend AI identifies unusual patterns. "
    "A high risk score does not confirm fraud."
)


# ---------------------------------------------------
# DASHBOARD SUMMARY
# ---------------------------------------------------

total_transactions = len(df)

high_risk_transactions = int(
    (df["risk_score"] > 60).sum()
)

medium_risk_transactions = int(
    (
        (df["risk_score"] > 30) &
        (df["risk_score"] <= 60)
    ).sum()
)

low_risk_transactions = int(
    (df["risk_score"] <= 30).sum()
)

average_risk_score = round(
    df["risk_score"].mean(),
    2
)


st.subheader("📌 Overall Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        total_transactions
    )

with col2:
    st.metric(
        "Low Risk",
        low_risk_transactions
    )

with col3:
    st.metric(
        "Medium Risk",
        medium_risk_transactions
    )

with col4:
    st.metric(
        "High Risk",
        high_risk_transactions
    )


st.metric(
    "Average Risk Score",
    average_risk_score
)


# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Transactions",
        "🚨 High Risk",
        "📊 Analytics",
        "ℹ️ About"
    ]
)


# ===================================================
# TAB 1 - TRANSACTIONS
# ===================================================

with tab1:

    st.subheader("📋 All Transactions")

    st.dataframe(
        df[
            [
                "transaction_id",
                "amount",
                "category",
                "hour",
                "location",
                "risk_score",
                "risk_level",
                "explanation"
            ]
        ],
        width="stretch"
    )


# ===================================================
# TAB 2 - HIGH RISK
# ===================================================

with tab2:

    st.subheader("🚨 Transactions Requiring Attention")

    high_risk_df = df[
        df["risk_score"] > 60
    ].sort_values(
        "risk_score",
        ascending=False
    )

    if len(high_risk_df) == 0:

        st.success(
            "No transactions currently have a risk score above 60."
        )

    else:

        st.dataframe(
            high_risk_df[
                [
                    "transaction_id",
                    "amount",
                    "category",
                    "hour",
                    "location",
                    "risk_score",
                    "risk_level",
                    "explanation"
                ]
            ],
            width="stretch"
        )


# ===================================================
# TAB 3 - ANALYTICS
# ===================================================

with tab3:

    st.subheader("📊 Risk Analytics")

    risk_distribution = (
        df["risk_level"]
        .value_counts()
    )

    st.bar_chart(
        risk_distribution,
        width="stretch"
    )

    st.subheader("Risk Score Trend")

    risk_chart = df[
        [
            "transaction_id",
            "risk_score"
        ]
    ].set_index(
        "transaction_id"
    )

    st.line_chart(
        risk_chart,
        width="stretch"
    )


# ===================================================
# TAB 4 - ABOUT
# ===================================================

with tab4:

    st.subheader("ℹ️ About SafeSpend AI")

    st.write(
        """
        SafeSpend AI is an intelligent transaction risk
        monitoring system designed to identify unusual
        transaction patterns.
        """
    )

    st.write("### Detection Methods")

    st.write(
        """
        • Z-Score
        • IQR
        • Isolation Forest
        • Local Outlier Factor (LOF)
        • Time-based anomaly detection
        """
    )

    st.write("### Risk Score")

    st.write(
        """
        The system combines multiple anomaly signals
        to produce a risk score between 0 and 100.
        """
    )

    st.write("### Important Note")

    st.warning(
        "An unusual transaction is not necessarily fraudulent. "
        "The system is intended to support review and monitoring."
    )


# ---------------------------------------------------
# FINAL DOWNLOAD
# ---------------------------------------------------

st.subheader("⬇️ Export Results")

final_csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Complete Results",
    data=final_csv,
    file_name="safespend_complete_results.csv",
    mime="text/csv",
    width="stretch"
)



# ===================================================
# SECTION 25 - COMPLETE APPLICATION TESTING
# ===================================================

st.header("🧪 Section 25 — Application Testing")

st.write(
    "Testing SafeSpend AI using different transaction scenarios."
)


# ---------------------------------------------------
# TEST SUMMARY
# ---------------------------------------------------

st.subheader("📋 Current System Test Summary")

test_total = len(df)

test_anomalies = int(
    (
        df["isolation_anomaly"] |
        df["lof_anomaly"] |
        df["time_anomaly"]
    ).sum()
)

test_high_risk = int(
    (df["risk_score"] > 60).sum()
)

test_valid_scores = bool(
    df["risk_score"].between(0, 100).all()
)

test_valid_levels = bool(
    df["risk_level"].isin(
        ["Low", "Medium", "High", "Very High"]
    ).all()
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions Tested",
        test_total
    )

with col2:
    st.metric(
        "Detected Anomalies",
        test_anomalies
    )

with col3:
    st.metric(
        "High Risk",
        test_high_risk
    )

with col4:
    if test_valid_scores:
        st.success("Scores Valid")
    else:
        st.error("Score Error")


# ---------------------------------------------------
# AUTOMATED CHECKS
# ---------------------------------------------------

st.subheader("✅ Automated Checks")

checks = {
    "Dataset is not empty": len(df) > 0,

    "Required transaction ID exists":
        "transaction_id" in df.columns,

    "Amount column exists":
        "amount" in df.columns,

    "Hour column exists":
        "hour" in df.columns,

    "Risk score between 0 and 100":
        test_valid_scores,

    "Risk levels are valid":
        test_valid_levels,

    "Isolation Forest results exist":
        "isolation_anomaly" in df.columns,

    "LOF results exist":
        "lof_anomaly" in df.columns,

    "Time anomaly results exist":
        "time_anomaly" in df.columns,

    "Explanation results exist":
        "explanation" in df.columns
}


test_results = pd.DataFrame(
    {
        "Test": list(checks.keys()),
        "Result": list(checks.values())
    }
)


st.dataframe(
    test_results,
    width="stretch"
)


# ---------------------------------------------------
# FINAL TEST RESULT
# ---------------------------------------------------

if all(checks.values()):

    st.success(
        "🎉 All automated application tests passed!"
    )

else:

    failed_tests = [
        name
        for name, result in checks.items()
        if not result
    ]

    st.error(
        "Some tests failed."
    )

    st.write(
        "Failed tests:",
        failed_tests
    )


# ---------------------------------------------------
# SAMPLE SCENARIO TESTS
# ---------------------------------------------------

st.subheader("🔬 Sample Transaction Scenarios")

scenario_data = pd.DataFrame({
    "Scenario": [
        "Normal Transaction",
        "High Amount",
        "Late Night",
        "Multiple Warning Signals"
    ],

    "Amount": [
        "₹500",
        "₹25,000",
        "₹1,200",
        "₹30,000"
    ],

    "Time": [
        "14:00",
        "14:00",
        "02:00",
        "01:00"
    ],

    "Expected Behaviour": [
        "Usually lower risk",
        "May receive higher risk",
        "May receive time anomaly",
        "May receive multiple anomaly signals"
    ]
})


st.dataframe(
    scenario_data,
    width="stretch"
)


st.info(
    "These scenarios demonstrate how the system can "
    "identify unusual patterns. They are testing examples "
    "and do not represent confirmed fraud."
)
