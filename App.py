import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# Page Configuration
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")
st.title("📊 Marketing Campaign Performance & Prediction Dashboard")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# Page Configuration
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")
st.title("📊 Marketing Campaign Performance & Prediction Dashboard")
@st.cache_data
def process_and_consolidate():

    # Load datasets
    tira = pd.read_csv(
        r"C:\Users\Nanda\OneDrive\Documents\Marketting_campaign_performance_prediction\Dataset\tira_campaign_data_with_nulls.csv",
        encoding_errors="ignore"
    )
    purple = pd.read_csv(
        r"C:\Users\Nanda\OneDrive\Documents\Marketting_campaign_performance_prediction\Dataset\purplle_campaign_data_with_nulls.csv",
        encoding_errors="ignore"
    )
    nyka = pd.read_csv(
        r"C:\Users\Nanda\OneDrive\Documents\Marketting_campaign_performance_prediction\Dataset\nykaa_campaign_data_with_nulls.csv",
        encoding_errors="ignore"
    )

    # Impute Tira
    numcols = tira.select_dtypes(include="number").columns
    tira[numcols] = tira[numcols].fillna(tira[numcols].median())
    catcol = tira.select_dtypes(exclude="number").columns
    for col in catcol:
        tira[col] = tira[col].fillna(tira[col].mode()[0])

    # Impute Purplle
    numeric = purple.select_dtypes(include="number").columns
    categorical = purple.select_dtypes(exclude="number").columns
    purple[numeric] = purple[numeric].fillna(purple[numeric].median())
    for col in categorical:
        purple[col] = purple[col].fillna(purple[col].mode()[0])

    # Impute Nykaa
    numericaldata = nyka.select_dtypes(include="number").columns
    categoricaldata = nyka.select_dtypes(exclude="number").columns
    nyka[numericaldata] = nyka[numericaldata].fillna(nyka[numericaldata].median())
    for col in categoricaldata:
        nyka[col] = nyka[col].fillna(nyka[col].mode()[0])

    # Consolidate
    consolidate = pd.concat([nyka, purple, tira], axis=0)
    consolidated1 = consolidate.copy(deep=True)

    # Feature Engineering
    consolidated1["CTR"] = consolidated1["Clicks"] / consolidated1["Leads"]
    consolidated1["leadRate"] = consolidated1["Leads"] / consolidated1["Clicks"]
    consolidated1["converstionrate"] = consolidated1["Conversions"] / consolidated1["Leads"]

    return consolidated1


df = process_and_consolidate()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚙️ Feature Encoding & Preprocessing",
    "📈 EDA & Correlation",
    "📉 Linear Regression",
    "🎯 Classification Models",
    "🌳 Decision Tree Rules",
    "🌲 Random Forest Classifier"
])
with tab1:
    st.subheader("Data Preprocessing & Feature Encoding Pipeline")

    st.markdown("""
    To prepare the raw transactional performance logs from **Tira, Purplle, and Nykaa** for advanced scikit-learn models, the following pipeline operations are automatically applied behind the scenes:
    """)

    col_enc1, col_enc2 = st.columns(2)

    with col_enc1:
        st.markdown("#### 🔹 Data Imputation Rules Implemented")
        st.markdown("""
        * **Numerical Variables:** Filled using the **Median**.
        * **Categorical Features:** Filled using the **Mode**.
        """)

        st.markdown("#### 🔹 Derived Engineered Features")
        st.code("""
df["CTR"] = df["Clicks"] / df["Leads"]
df["leadRate"] = df["Leads"] / df["Clicks"]
df["converstionrate"] = df["Conversions"] / df["Leads"]
        """, language="python")

    with col_enc2:
        st.markdown("#### 🔹 Label Encoding Transformation Summary")

        encoding_preview = df[["Campaign_Type", "Target_Audience", "Customer_Segment"]].copy()
        for col in encoding_preview.columns:
            encoding_preview[col] = LabelEncoder().fit_transform(encoding_preview[col])

        st.dataframe(encoding_preview.head(6))

    st.markdown("---")
    st.markdown("#### 📋 Consolidated Dataset Sample Preview")
    st.dataframe(df.head(5))
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Target Variable Revenue Correlation Matrix**")
        corr_cols = ["Clicks", "Leads", "Conversions", "Revenue", "Impressions", "ROI",
                     "Engagement_Score", "Duration", "Acquisition_Cost"]
        correlation = df[corr_cols].corr()["Revenue"]
        st.dataframe(correlation)

    with col2:
        st.markdown("**Conversions vs Revenue Trend**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            data=df,
            x='Conversions',
            y='Revenue',
            ci=None,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'},
            ax=ax
        )
        st.pyplot(fig)
with tab3:
    st.subheader("Linear Regression Pipeline: Revenue Prediction")

    features_list = ["Clicks", "Leads", "Conversions", "ROI", "Engagement_Score", "Impressions"]
    X_lr = df[features_list]
    y_lr = df[["Revenue"]]

    x_train, x_test, y_train, y_test = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=4
    )

    lrmodel = Pipeline([
        ("scalar", MinMaxScaler()),
        ("linear", LinearRegression())
    ])

    lrmodel.fit(x_train, y_train)
    y_pred = lrmodel.predict(x_test)
    lr_score = r2_score(y_test, y_pred)

    st.metric(label="Linear Regression R² Variance Score", value=f"{lr_score:.6f}")

    with st.expander("Model Intercept and Scaled Coefficients"):
        st.write("Intercept:", lrmodel.named_steps['linear'].intercept_)

        coef_df = pd.DataFrame({
            'Feature': features_list,
            'Coefficient': lrmodel.named_steps['linear'].coef_.flatten()
        })

        st.dataframe(coef_df)
with tab4:
    st.subheader("Binary Profit Margin Classifiers")

    logistic_copy = df.copy(deep=True)
    logistic_copy = logistic_copy.drop(["Campaign_ID", "Date"], axis=1, errors='ignore')

    logistic_copy["Profit"] = logistic_copy["Revenue"] - logistic_copy["Acquisition_Cost"]
    logistic_copy["Profit_Confirm"] = (logistic_copy["Profit"] > 1.00).astype(int)

    X_log = logistic_copy[["Revenue", "Acquisition_Cost", "Conversions"]]
    y_log = logistic_copy["Profit_Confirm"]

    log_xtrain, log_xtest, log_ytrain, log_ytest = train_test_split(
        X_log, y_log, test_size=0.9, random_state=42
    )

    c1, c2 = st.columns(2)

    # ---------------- Logistic Regression ----------------
    with c1:
        st.markdown("### 1. Logistic Regression Model")

        log_model = Pipeline([
            ("logistic", LogisticRegression())
        ])

        log_model.fit(log_xtrain, log_ytrain)
        pred_log = log_model.predict(log_xtest)

        acc_log = accuracy_score(log_ytest, pred_log)
        st.metric(label="Logistic Accuracy", value=f"{acc_log:.6f}")

        st.markdown("**Classification Report:**")
        st.text(classification_report(log_ytest, pred_log))

        st.markdown("**Confusion Matrix:**")
        st.write(confusion_matrix(log_ytest, pred_log))

    # ---------------- XGBoost Classifier ----------------
    with c2:
        st.markdown("### 2. Tuned XGBoost Classifier Model")

        pipeline_xgb = Pipeline([
            ('scalar', MinMaxScaler()),
            ('xgb', XGBClassifier(
                subsample=0.7,
                n_estimators=300,
                min_child_weight=3,
                max_depth=3,
                learning_rate=0.1,
                gamma=0.2,
                colsample_bytree=0.7,
                random_state=42
            ))
        ])

        xgb_features = logistic_copy[["Campaign_Type", "Target_Audience", "Customer_Segment"]].copy()
        for col in xgb_features.columns:
            xgb_features[col] = LabelEncoder().fit_transform(xgb_features[col])

        xgb_features["Profit"] = logistic_copy["Profit"]
        xgb_target = (logistic_copy["Profit"] > logistic_copy["Profit"].mean()).astype(int)

        dtrain, dtest, ttrain, ttest = train_test_split(
            xgb_features, xgb_target, test_size=0.2, random_state=4, stratify=xgb_target
        )

        pipeline_xgb.fit(dtrain, ttrain)
        xgb_pred = pipeline_xgb.predict(dtest)

        acc_xgb = accuracy_score(ttest, xgb_pred)
        st.metric(label="XGBoost Target Accuracy", value=f"{acc_xgb:.6f}")

        st.markdown("Classification Report:")
        st.text(classification_report(ttest, xgb_pred))

        st.markdown("Confusion Matrix:")
        st.write(confusion_matrix(ttest, xgb_pred))
with tab5:
    st.subheader("Decision Tree Classification Flow")

    dt_copy = df.copy(deep=True)
    dt_copy["Profit"] = dt_copy['Revenue'] - dt_copy['Acquisition_Cost']
    profit_mean = dt_copy["Profit"].mean()

    st.info(f"Target Baseline split threshold configured around Mean Profit Margin: {profit_mean:.4f}")

    dt_trimmed = dt_copy[["Campaign_Type", "Target_Audience", "Customer_Segment"]].copy()
    for col in dt_trimmed.columns:
        dt_trimmed[col] = LabelEncoder().fit_transform(dt_trimmed[col])

    dt_trimmed["Profit"] = dt_copy["Profit"]
    dt_target = (dt_copy["Profit"] > profit_mean).astype(int)

    dtrain_dt, dtest_dt, ttrain_dt, ttest_dt = train_test_split(
        dt_trimmed, dt_target, test_size=0.2, random_state=4, stratify=dt_target
    )

    dtree_model = Pipeline([
        ("scalar", MinMaxScaler()),
        ("decision", DecisionTreeClassifier(
            splitter='best',
            min_samples_split=5,
            min_samples_leaf=2,
            max_depth=5,
            criterion='gini',
            random_state=42
        ))
    ])

    dtree_model.fit(dtrain_dt, ttrain_dt)
    tpred_dt = dtree_model.predict(dtest_dt)

    col_dt1, col_dt2 = st.columns(2)

    with col_dt1:
        st.metric(label="Decision Tree Prediction Accuracy",
                  value=f"{accuracy_score(ttest_dt, tpred_dt):.6f}")
        st.markdown("Confusion Matrix:")
        st.write(confusion_matrix(ttest_dt, tpred_dt))

    with col_dt2:
        st.markdown("Classification Metrics Evaluation:")
        st.text(classification_report(ttest_dt, tpred_dt))
with tab6:
    st.subheader("Random Forest Ensemble Classification Performance")

    rf_copy = df.copy(deep=True)
    rf_copy["Profit"] = rf_copy['Revenue'] - rf_copy['Acquisition_Cost']
    rf_profit_mean = rf_copy["Profit"].mean()

    rf_trimmed = rf_copy[["Campaign_Type", "Target_Audience", "Customer_Segment"]].copy()
    for col in rf_trimmed.columns:
        rf_trimmed[col] = LabelEncoder().fit_transform(rf_trimmed[col])

    rf_trimmed["Profit"] = rf_copy["Profit"]
    rf_target = (rf_copy["Profit"] > rf_profit_mean).astype(int)

    dtrain_rf, dtest_rf, ttrain_rf, ttest_rf = train_test_split(
        rf_trimmed, rf_target, test_size=0.2, random_state=4, stratify=rf_target
    )

    rf_model = Pipeline([
        ("forest", RandomForestClassifier(n_estimators=100, random_state=4))
    ])

    rf_model.fit(dtrain_rf, ttrain_rf.values.ravel())
    rf_pred = rf_model.predict(dtest_rf)

    col_rf1, col_rf2 = st.columns(2)

    with col_rf1:
        st.metric(label="Random Forest Accuracy Score",
                  value=f"{accuracy_score(ttest_rf, rf_pred):.6f}")
        st.markdown("Confusion Matrix:")
        st.write(confusion_matrix(ttest_rf, rf_pred))

    with col_rf2:
        st.markdown("Classification Report Evaluation:")
        st.text(classification_report(ttest_rf, rf_pred))
