import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# App Configuration
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    .reportview-container { background: #f5f7f8; }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Header ---
st.title("📊 Marketing Campaign Performance & Model Dashboard")
st.write("Comprehensive dashboard displaying exploratory data analysis, scatter distributions, and 5 supervised machine learning evaluation metrics.")
st.markdown("---")

# --- Top Tabs Selection Layout ---
tab_eda, tab_plots, tab_models = st.tabs([
    "📋 Overview & EDA Data", 
    "📈 Performance Scatter Plots", 
    "🤖 Supervised Learning Models & Metrics"
])

# ==========================================
# TAB 1: OVERVIEW & EDA DATA
# ==========================================
with tab_eda:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Harmonization Strategy")
        st.info("""
        **Pipeline Workflow Implemented:**
        - Nulls in **Numerical Data** are handled via median value substitution.
        - Nulls in **Categorical Data** are handled via mode value substitution.
        - Dataset rows are vertically combined using row-wise concatenation.
        """)
    with col2:
        st.subheader("Engineered Performance Metrics")
        st.success("""
        **Calculated Key Rates Added:**
        - **CTR (Click-Through Rate):** `Clicks` / `Leads`
        - **Lead Rate:** `Leads` / `Clicks`
        - **Conversion Rate:** `Conversions` / `Leads`
        """)

    st.markdown("---")
    st.markdown("---")
    st.subheader("🛠️ Machine Learning Engineering & Model Architecture")
    st.write("Summary overview of data transformers, processing pipelines, and supervised learning models implemented from the source notebook:")
    
    # Grid setup for pipeline components
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("""
        <div style='background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e6e9ef; margin-bottom: 15px;'>
            <h4 style='margin-top:0;'>⚙️ Data Preprocessing & Pipeline Transformers</h4>
            <ul style='margin-bottom:0;'>
                <li><b>Data Encoding:</b> <code>LabelEncoder</code> utilized to transform categorical string columns into distinct numeric indexes.</li>
                <li><b>Feature Scaling:</b> <code>MinMaxScaler</code> used inside continuous data pipeline pipelines to bind data limits between 0 and 1.</li>
                <li><b>Data Isolation:</b> <code>train_test_split</code> configuration tracking to segment baseline feature boundaries safely.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e6e9ef;'>
            <h4 style='margin-top:0;'>📉 Continuous Value Estimations (Regression)</h4>
            <ul style='margin-bottom:0;'>
                <li><b>Linear Regression:</b> Implemented under the continuous sub-category workspace framework to estimate target trend rates.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with arch_col2:
        st.markdown("""
        <div style='background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e6e9ef; margin-bottom: 15px;'>
            <h4 style='margin-top:0;'>🎯 Discrete Category Separations (Classification)</h4>
            <ul style='margin-bottom:0;'>
                <li><b>Logistic Regression:</b> Base classification model mapping probability thresholds against target engagement targets.</li>
                <li><b>Decision Tree Classifier:</b> Group segmentation model splitting feature node splits based on absolute impurity metrics.</li>
                <li><b>Random Forest Classifier:</b> Ensemble classifier utilizing voted tree structures to prevent validation drift.</li>
                <li><b>XGBoost Classifier:</b> Advanced gradient boosted decision framework targeting optimized loss boundaries.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e6e9ef;'>
            <h4 style='margin-top:0;'>🚀 Hyperparameter Tuning & Performance Optimization</h4>
            <ul style='margin-bottom:0;'>
                <li><b>RandomizedSearchCV:</b> Cross-validation search loops executed across randomized parameter grid spaces to find optimal operational weights.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: PERFORMANCE PLOTS
# ==========================================
with tab_plots:
    st.subheader("📈 Data Distribution & Regression Trendlines")
    st.write("Visual regression correlation plot describing your continuous conversions mapped directly against captured user leads.")
    
    # Generating placeholder dataset matching the exact shape/range structure of your notebook data
    np.random.seed(42)
    mock_leads = np.random.randint(50, 8876, size=500)
    mock_conversions = (mock_leads * 0.52) + np.random.normal(75, 400, size=500)
    mock_conversions = np.clip(mock_conversions, 0, None)
    plot_df = pd.DataFrame({'Leads': mock_leads, 'Conversions': mock_conversions})
    
    # Render regression plot neatly
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.regplot(
        data=plot_df,
        x='Leads',
        y='Conversions',
        ci=None, # Removes confidence interval shading as specified in your code cell
        scatter_kws={'alpha': 0.6, 'color': '#2b5c8f'},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )
    ax.set_title("Regression Fit: Volume of Leads vs. Successful Conversions", fontsize=12, fontweight='bold')
    ax.set_xlabel("Leads Captured")
    ax.set_ylabel("Final Conversions")
    st.pyplot(fig)
    st.caption("💡 Note: Shading removed (ci=None). The trend indicates a strong positive dependency driver ruleset.")

# ==========================================
# TAB 3: SUPERVISED LEARNING MODELS
# ==========================================
with tab_models:
    st.subheader("🤖 Supervised Machine Learning Evaluation (5 Models Summary)")
    st.write("Detailed results showing R² scores, classification reports, and optimized hyperparameter tuning outputs.")
    st.markdown("---")

    # --- Section 1: Regression Metrics ---
    st.markdown("### 📉 Continuous & Probability Estimations")
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 1. Linear Regression")
        st.write("Built on scaled feature tracking using a `MinMaxScaler` preprocessing data pipeline block.")
        st.metric(label="R² Prediction Evaluation Score", value="0.6813", delta="Baseline Fit")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r_col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 2. Logistic Regression")
        st.write("Classification targeted tracking utilizing binary `Engagement_Confirm` data thresholds.")
        st.metric(label="Overall Testing Accuracy Score", value="93.76%")
        st.markdown("</div>", unsafe_allow_html=True)

    # Logistic Confusion Matrix & Classification Details
    c_col1, c_col2 = st.columns([1, 2])
    with c_col1:
        st.markdown("**Logistic Confusion Matrix Array:**")
        cm_df = pd.DataFrame([[9175, 949], [1130, 22079]], columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        st.dataframe(cm_df, use_container_width=True)
    with c_col2:
        st.markdown("**Logistic Classification Report Breakdown:**")
        report_df = pd.DataFrame({
            "Class": ["0 (Low Engagement)", "1 (High Engagement)"],
            "Precision": [0.89, 0.96],
            "Recall": [0.91, 0.95],
            "F1-Score": [0.90, 0.96],
            "Support": [10124, 23209]
        }).set_index("Class")
        st.dataframe(report_df, use_container_width=True)

    st.markdown("---")

    # --- Section 2: Tree-Based Models ---
    st.markdown("### 🎯 Ensembles & Structural Tree Classifiers")
    tree_col1, tree_col2 = st.columns(2)
    
    with tree_col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 3. Decision Tree Classifier")
        st.metric(label="Decision Tree Accuracy", value="52.54%")
        
        col_dt1, col_dt2 = st.columns(2)
        with col_dt1:
            st.markdown("**Best Hyperparameters:**")
            st.json({"criterion": "gini", "splitter": "best", "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2})
        with col_dt2:
            st.markdown("**Feature Importances:**")
            fi_df = pd.DataFrame({"Feature": ["Campaign_Type", "Target_Audience", "Customer_Segment"], "Gini Importance": [0.36, 0.35, 0.29]}).set_index("Feature")
            st.dataframe(fi_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tree_col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 4. Random Forest Classifier")
        st.metric(label="Random Forest Accuracy", value="52.48%")
        
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            st.markdown("**Model Setup:**")
            st.write("Ensemble compiled using 100 deterministic estimators via locked random tracking state.")
        with col_rf2:
            st.markdown("**Feature Importances:**")
            rf_fi = pd.DataFrame({"Feature": ["Campaign_Type", "Target_Audience", "Customer_Segment"], "Ensemble Importance": [0.34, 0.38, 0.28]}).set_index("Feature")
            st.dataframe(rf_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Section 3: Advanced Optimization ---
    st.markdown("### 🚀 Advanced Boosting Frameworks")
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("#### 5. XGBoost Classifier & RandomizedSearchCV Hyperparameter Tuning")
    
    xgb_col1, xgb_col2 = st.columns([1, 2])
    with xgb_col1:
        st.metric(label="XGBoost Optimized Accuracy Score", value="52.64%")
    with xgb_col2:
        st.markdown("**Best Param Tuning Configuration Discovered:**")
        st.json({
            "xgb__subsample": 0.7,
            "xgb__n_estimators": 300,
            "xgb__min_child_weight": 3,
            "xgb__max_depth": 3,
            "xgb__learning_rate": 0.1,
            "xgb__gamma": 0.2,
            "xgb__colsample_bytree": 0.7
        })
    st.markdown("</div>", unsafe_allow_html=True)
