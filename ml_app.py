import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

# Configuration
st.set_page_config(layout="wide")

# Seaborn Datasets
seaborn_datasets = ["iris", "tips", "penguins", "titanic", "car_crashes"]

# Helper Functions
@st.cache_data 
def load_data(source_option, uploaded_file=None, selected_dataset=None):
    """Loads data based on user selection."""
    try:
        if source_option == "Seaborn Dataset" and selected_dataset:
            if selected_dataset in seaborn_datasets:
                df = sns.load_dataset(selected_dataset); st.sidebar.success(f"Loaded '{selected_dataset}'."); return df
            else: st.sidebar.error(f"Invalid dataset: {selected_dataset}"); return None
        elif source_option == "Upload CSV" and uploaded_file is not None:
            try: df = pd.read_csv(uploaded_file); st.sidebar.success("CSV loaded."); return df
            except Exception as csv_e: st.sidebar.error(f"Error reading CSV: {csv_e}"); return None
        else: return None
    except Exception as e: st.sidebar.error(f"Error loading data: {e}"); return None

# App Layout & Sidebar
st.title("📊 Simple ML Model Trainer")
st.sidebar.header("1. Data Loading")
data_source = st.sidebar.radio("Data source:", ("Seaborn Dataset", "Upload CSV"), key="data_source_radio")
df = None
if data_source == "Seaborn Dataset":
    default_idx = seaborn_datasets.index("iris") if "iris" in seaborn_datasets else 0
    selected_dataset = st.sidebar.selectbox("Select Seaborn Dataset", seaborn_datasets, index=default_idx)
    if selected_dataset: df = load_data(data_source, selected_dataset=selected_dataset)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file: df = load_data(data_source, uploaded_file=uploaded_file)

# Main Area Processing
if df is not None:
    st.header("1. Data Preview"); st.dataframe(df.head())
    # Data cleaning
    rows_before = df.shape[0]; df_cleaned = df.dropna().copy(); rows_after = df_cleaned.shape[0]
    if rows_before > rows_after: st.warning(f"Dropped {rows_before - rows_after} NA rows."); st.dataframe(df_cleaned.head())
    if df_cleaned.empty: st.error("No data left after dropping NAs."); st.stop()
    df = df_cleaned

    st.header("2. Configuration")
    col_config1, col_config2 = st.columns(2)
    # --- Configuration Form ---
    with st.form("ml_config_form"):
        with col_config1: 
            st.subheader("Feature Selection")
            all_columns = df.columns.tolist(); default_target_index = len(all_columns) - 1
            common_targets = ['species','survived','target','class','tip','price','mpg','total_bill','method','number','condition']
            for target in reversed(common_targets):
                if target in all_columns: default_target_index = all_columns.index(target); break
            target_var = st.selectbox("Target Variable (y)", all_columns, index=default_target_index)

            numeric_cols=df.select_dtypes(include=np.number).columns.tolist(); categorical_cols=df.select_dtypes(exclude=np.number).columns.tolist()
            available_numeric = [c for c in numeric_cols if c != target_var]; available_categorical = [c for c in categorical_cols if c != target_var]
            selected_numeric = st.multiselect("Quantitative Features (X)", available_numeric, default=available_numeric[:min(len(available_numeric), 5)])
            selected_categorical = st.multiselect("Qualitative Features (X)", available_categorical, default=available_categorical[:min(len(available_categorical), 3)])
            selected_features = selected_numeric + selected_categorical

            task_type = None
            if target_var:
                target_series = df[target_var]; nunique = target_series.nunique()
                if pd.api.types.is_numeric_dtype(target_series.dtype): task_type = "Classification" if nunique <= 15 or pd.api.types.is_integer_dtype(target_series.dtype) else "Regression"
                else: task_type = "Classification" if nunique <= 50 else None;
                if task_type == "Classification" and nunique > 50: st.warning(">50 unique categories.")
                if task_type: st.info(f"Detected Task: **{task_type}**")
            else: st.warning("Select target.")
        with col_config2: # Model/Parameters
            st.subheader("Model Selection & Parameters")
            model_options = []; model_type = None
            if task_type == "Regression": model_options = ["Linear Regression", "Random Forest Regressor"]
            elif task_type == "Classification":
                model_options = ["Logistic Regression", "Random Forest Classifier"]
                if target_var and df[target_var].nunique() > 2: model_options = ["Random Forest Classifier"]; st.info(">2 classes, using RF.")
            if model_options: model_type = st.selectbox("Select Model", model_options)
            else: st.warning("No models available.")

            test_size = st.slider("Test Size (%)", 10, 50, 25, 5) / 100.0
            n_estimators, max_depth = 100, None
            if model_type and "Random Forest" in model_type:
                n_estimators = st.slider("Num Estimators (RF)", 10, 300, 100, 10)
                max_depth = st.slider("Max Depth (RF)", 1, 30, 10, 1) if st.checkbox("Limit Max Depth", True) else None
        submitted = st.form_submit_button("🚀 Fit Model & Evaluate")

    # Model Training & Results
    if submitted:
        if not target_var or not selected_features or task_type is None or not model_type:
            st.error("Config incomplete."); st.stop()

        st.header("3. Results")
        try:
            # --- 1. Preprocessing ---
            X = df[selected_features].copy(); y = df[target_var].copy()
            if selected_categorical: X = pd.get_dummies(X, columns=selected_categorical, drop_first=True, dtype=int)
            final_feature_names = X.columns.tolist()
            label_classes, le = None, None
            if task_type == "Classification":
                if not pd.api.types.is_numeric_dtype(y): le = LabelEncoder(); y = le.fit_transform(y); label_classes = le.classes_
                else: label_classes = np.sort(y.unique())
                if not isinstance(y, pd.Series): y = pd.Series(y, name=target_var, index=X.index)

            # Stratification & Split
            stratify_param = y if task_type == 'Classification' and not y.empty and y.value_counts().min() >= 2 else None
            if task_type == 'Classification' and stratify_param is None and not y.empty: st.warning("Cannot stratify: min class count < 2.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_param)
            st.write(f"Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

            # 2. Model Instantiation
            model = None 
            if model_type == "Linear Regression": model = LinearRegression()
            elif model_type == "Random Forest Regressor": model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            elif model_type == "Logistic Regression":
                if y.nunique() != 2: st.error(f"LogReg needs binary target, found {y.nunique()}."); st.stop()
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "Random Forest Classifier": model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else: st.error("Model type not recognized."); st.stop() 

            # 3. Training
            with st.spinner(f"Training {model_type}..."): model.fit(X_train, y_train)
            st.success(f"Model '{model_type}' trained!")

            # 4. Prediction
            y_pred = model.predict(X_test); y_prob = None
            is_binary_classification = False
            if task_type == "Classification":
                # We check it based on original unique classes
                is_binary_classification = len(np.unique(y)) == 2 
                if hasattr(model, "predict_proba") and is_binary_classification:
                     # We need try-except as predict_proba might still fail
                     try: 
                         y_prob = model.predict_proba(X_test)[:, 1]
                     except Exception as proba_err:
                         st.warning(f"Could not get probabilities for ROC: {proba_err}")
                         # We want to ensure it's None if it fails
                         y_prob = None 

            # 5. Metrics
            st.subheader("Performance Metrics")
            results_col1, results_col2 = st.columns(2) 
            with results_col1:
                if task_type == "Regression":
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
                    st.metric("R2", f"{r2_score(y_test, y_pred):.4f}")
                elif task_type == "Classification":
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            with results_col2:
                 if task_type == "Classification":
                     # We are going to display Confusion Matrix data here
                     cm = confusion_matrix(y_test, y_pred)
                     display_labels = label_classes if label_classes is not None else getattr(model, 'classes_', None)
                     st.write("**Confusion Matrix (Data):**")
                     if display_labels is not None:
                        labels_short = [str(lbl)[:15]+"..." if len(str(lbl))>15 else str(lbl) for lbl in display_labels]
                        try: st.dataframe(pd.DataFrame(cm, index=labels_short, columns=labels_short))
                        except: st.dataframe(pd.DataFrame(cm))
                     else: st.dataframe(pd.DataFrame(cm))


            # 6. Visualizations
            st.subheader("Visualizations")

            if task_type == "Regression":
                # Regression Plots
                plot_col_reg1, plot_col_reg2 = st.columns(2)
                with plot_col_reg1:
                    st.write("**Residual Distribution**")
                    fig_res, ax_res = plt.subplots(figsize=(6, 4))
                    sns.histplot(y_test - y_pred, kde=True, ax=ax_res)
                    ax_res.set_title("Residual Distribution"); ax_res.set_xlabel("Residuals")
                    plt.tight_layout(); st.pyplot(fig_res); plt.close(fig_res)
                with plot_col_reg2:
                    st.write("**Feature Importance**")
                    fig_imp_reg, ax_imp_reg = plt.subplots(figsize=(6, 4))
                    importance_df_reg = None; imp_col_name_reg = 'Importance'
                    try:
                        if hasattr(model, 'feature_importances_'): imp = model.feature_importances_
                        elif hasattr(model, 'coef_'): imp = abs(model.coef_[0] if model.coef_.ndim>1 else model.coef_); imp_col_name_reg = 'Importance (Abs Coef)'
                        else: imp = None
                        if imp is not None: importance_df_reg = pd.DataFrame({'Feature': final_feature_names, imp_col_name_reg: imp}).sort_values(imp_col_name_reg, ascending=False)

                        if importance_df_reg is not None:
                             top_n = min(len(importance_df_reg), 15)
                             sns.barplot(x=imp_col_name_reg, y='Feature', data=importance_df_reg.head(top_n), ax=ax_imp_reg, palette="viridis")
                             ax_imp_reg.set_title(f"Top {top_n} Features"); plt.tight_layout(); st.pyplot(fig_imp_reg)
                        else: st.info("Feature importance not available.")
                    except Exception as e: st.warning(f"Could not plot importance: {e}")
                    finally: plt.close(fig_imp_reg)

            elif task_type == "Classification":
                # Classification Plots, We are going to Attempt all 3
                st.write("---") 
                plot_col_clf1, plot_col_clf2, plot_col_clf3 = st.columns(3)

                # Plot 1: Confusion Matrix Heatmap
                with plot_col_clf1:
                    st.write("**Confusion Matrix Heatmap**")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) 
                    try:
                        cm = confusion_matrix(y_test, y_pred)
                        display_labels = label_classes if label_classes is not None else getattr(model, 'classes_', None)
                        labels_short = [str(lbl)[:8]+"..." if len(str(lbl))>8 else str(lbl) for lbl in display_labels] if display_labels is not None else 'auto'
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=labels_short, yticklabels=labels_short)
                        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True"); ax_cm.set_title("Confusion Matrix")
                        plt.tight_layout(); st.pyplot(fig_cm)
                    except Exception as e: st.warning(f"Could not plot CM: {e}")
                    finally: plt.close(fig_cm)

                # Plot 2: Feature Importance
                with plot_col_clf2:
                    st.write("**Feature Importance**")
                    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
                    importance_df = None; imp_col_name = 'Importance'
                    plot_imp_success = False
                    try:
                        if hasattr(model, 'feature_importances_'): imp = model.feature_importances_
                        elif hasattr(model, 'coef_'):
                            if len(model.classes_)==2: 
                                imp = abs(model.coef_[0]); imp_col_name = 'Importance (Abs Coef)'
                            else: imp = None 
                        else: imp = None
                        if imp is not None: importance_df = pd.DataFrame({'Feature': final_feature_names, imp_col_name: imp}).sort_values(imp_col_name, ascending=False)

                        if importance_df is not None:
                            top_n = min(len(importance_df), 15)
                            sns.barplot(x=imp_col_name, y='Feature', data=importance_df.head(top_n), ax=ax_imp, palette="viridis")
                            ax_imp.set_title(f"Top {top_n} Features"); plt.tight_layout(); st.pyplot(fig_imp); plot_imp_success = True

                    except Exception as e: st.warning(f"Could not plot importance: {e}")
                    finally:
                        if not plot_imp_success: st.info("Feature importance not available for this model.") # Show message if plot failed or df is None
                        plt.close(fig_imp)

                # Plot 3: ROC Curve 
                with plot_col_clf3:
                    st.write("**ROC Curve**")
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    plot_roc_success = False
                    try:
                        # We are now going to check if binary and probabilities exist
                        if is_binary_classification and y_prob is not None:
                            fpr, tpr, _ = roc_curve(y_test, y_prob); roc_auc = auc(fpr, tpr)
                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05]); ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.set_title('ROC Curve'); ax_roc.legend(loc="lower right")
                            plt.tight_layout(); st.pyplot(fig_roc); plot_roc_success = True

                    except Exception as e: st.warning(f"Could not plot ROC: {e}")
                    finally:
                        if not plot_roc_success:
                             if not is_binary_classification: st.info("ROC curve visualization is typically shown for binary classification tasks.")
                             elif y_prob is None: st.info("ROC curve could not be generated (predict_proba unavailable or failed).")
                             else: st.info("ROC Curve could not be generated.") # Generic fallback
                        plt.close(fig_roc)

            # 7. Model Export
            st.subheader("Model Export")
            try:
                model_buffer = BytesIO(); joblib.dump(model, model_buffer); model_buffer.seek(0)
                data_name = selected_dataset if data_source == "Seaborn Dataset" else "uploaded_data"
                model_filename = f"{data_name}_{model_type.replace(' ', '_')}_model.joblib"
                st.download_button(label="Download Trained Model (.joblib)", data=model_buffer, file_name=model_filename, mime="application/octet-stream")
            except Exception as export_err: st.warning(f"Could not prepare model for download: {export_err}")

        # Error Handling
        except Exception as e: st.error(f"An error occurred: {e}"); st.exception(e)

# Fallback message
elif df is None and not (data_source=="Upload CSV" and uploaded_file is None):
    st.info("Load data to begin.")
