import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder # Added for potential categorical target handling
import matplotlib.pyplot as plt
import io 

# --- Configuration ---
st.set_page_config(layout="wide") 
# The line below might cause errors on some Streamlit versions and is not essential
# st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Helper Functions ---

@st.cache_data # Cache the data loading
def load_data(source_option, uploaded_file=None, selected_dataset=None):
    """Loads data based on user selection."""
    try:
        if source_option == "Seaborn Dataset" and selected_dataset:
            # Ensure selected_dataset is valid before loading
            valid_seaborn_datasets = ["iris", "tips", "penguins", "diamonds", "mpg"]
            if selected_dataset in valid_seaborn_datasets:
                df = sns.load_dataset(selected_dataset)
                st.sidebar.success(f"Loaded '{selected_dataset}' dataset.")
                return df
            else:
                st.sidebar.error(f"Invalid dataset selected: {selected_dataset}")
                return None
        elif source_option == "Upload CSV" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file uploaded successfully.")
            return df
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None

def get_feature_names_after_dummies(df, selected_categorical_features):
    """Gets feature names including those created by get_dummies."""
    # Create a temporary df with selected features to apply get_dummies
    temp_df = df.copy()
    # Ensure only selected categorical features are processed if any exist
    cols_to_encode = [col for col in selected_categorical_features if col in temp_df.columns]
    if cols_to_encode:
         # prefix_sep ensures we can reliably split later if needed, but we use columns attr here
        temp_df_dummies = pd.get_dummies(temp_df[cols_to_encode], drop_first=True, dtype=int, prefix_sep='_')
        # Drop original categorical columns and join dummies
        temp_df = temp_df.drop(columns=cols_to_encode)
        temp_df = pd.concat([temp_df, temp_df_dummies], axis=1)

    # Return all column names (original numeric + new dummies)
    return temp_df.columns.tolist()

st.title("📊 Simple ML Model Trainer")

st.sidebar.header("1. Data Loading")
data_source = st.sidebar.radio(
    "Select data source:",
    ("Seaborn Dataset", "Upload CSV"),
    key="data_source_radio"
)

df = None # Initialize df
uploaded_file = None
selected_dataset = None


seaborn_datasets = ["iris", "tips", "penguins", "diamonds", "mpg"]

if data_source == "Seaborn Dataset":
    # Use the predefined list instead of sns.get_dataset_names()
    selected_dataset = st.sidebar.selectbox(
        "Select Seaborn Dataset",
        seaborn_datasets,
        index=0 # Default to the first dataset ('iris')
    )
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data using the helper function
df = load_data(data_source, uploaded_file, selected_dataset)

# --- Main Area ---
if df is not None:
    st.header("1. Data Preview")
    st.dataframe(df.head())

    # --- Simple Data Cleaning (Optional but recommended) ---
    # Drop rows with any missing values for simplicity in this basic app
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    if rows_before > rows_after:
        st.warning(f"Dropped {rows_before - rows_after} rows containing missing values for simplicity.")
        st.dataframe(df.head()) # Show preview again if rows were dropped


    st.header("2. Configuration")
    # Use columns for better layout within the form
    col_config1, col_config2 = st.columns(2)

    # --- Configuration Form ---
    with st.form("ml_config_form"):
        with col_config1:
            st.subheader("Feature Selection")
            # Identify potential target variables (can be numeric or categorical)
            all_columns = df.columns.tolist()
            # Try to make a sensible default target guess
            default_target_index = len(all_columns) - 1 if all_columns else 0
            if 'species' in all_columns: # common target in iris/penguins
                default_target_index = all_columns.index('species')
            elif 'tip' in all_columns: # common target in tips
                 default_target_index = all_columns.index('tip')
            elif 'price' in all_columns: # common target in diamonds
                 default_target_index = all_columns.index('price')
            elif 'mpg' in all_columns: # common target in mpg
                 default_target_index = all_columns.index('mpg')


            target_var = st.selectbox(
                "Select Target Variable (y)",
                all_columns,
                index=default_target_index
             )

            # Separate features based on type for selection widgets
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            # Remove target variable from potential features
            available_numeric_features = [col for col in numeric_cols if col != target_var]
            available_categorical_features = [col for col in categorical_cols if col != target_var]

            selected_numeric_features = st.multiselect(
                "Select Quantitative Features (X)",
                available_numeric_features,
                default=available_numeric_features # Default to all numeric initially
            )
            selected_categorical_features = st.multiselect(
                "Select Qualitative Features (X)",
                available_categorical_features,
                default=available_categorical_features # Default to all categorical initially
            )
            selected_features = selected_numeric_features + selected_categorical_features

            # Simple check for task type based on target variable dtype
            task_type = None # Initialize
            if target_var and target_var in df.columns:
                target_dtype = df[target_var].dtype
                unique_values_count = df[target_var].nunique()

                if pd.api.types.is_numeric_dtype(target_dtype):
                     # Heuristic: If numeric but has few unique values (e.g., <= 15) OR is integer, treat as classification
                    if unique_values_count <= 15 or pd.api.types.is_integer_dtype(target_dtype):
                         task_type = "Classification"
                    else:
                         task_type = "Regression"
                else: # Object, category, boolean types
                    task_type = "Classification"

                st.info(f"Detected Task Type: **{task_type}** (based on target '{target_var}')")
            else:
                st.warning("Select a target variable to determine task type.")

        with col_config2:
            st.subheader("Model Selection & Parameters")

            # Filter models based on detected task type
            if task_type == "Regression":
                model_options = ["Linear Regression", "Random Forest Regressor"]
            elif task_type == "Classification":
                # Check number of unique classes for Logistic Regression viability
                if target_var and df[target_var].nunique() > 2:
                    # Allow multi-class for Random Forest, but maybe not basic Logistic Regression setup
                     model_options = ["Random Forest Classifier"]
                     st.info("Target has more than 2 classes. Only Random Forest Classifier is available.")
                elif target_var: # Binary classification
                     model_options = ["Logistic Regression", "Random Forest Classifier"]
                else: # target_var not selected yet
                     model_options = ["Logistic Regression", "Random Forest Classifier"]
            else:
                model_options = [] # No models if task type unknown

            # Ensure default model is valid
            default_model_index = 0
            if not model_options:
                 st.warning("No models available for the current configuration.")
                 model_type = None
            else:
                 model_type = st.selectbox("Select Model", model_options, index=default_model_index)


            test_size = st.slider("Test Set Size (%)", 10, 50, 25, 5) / 100.0

            # Model-specific parameters
            n_estimators = 100 # Default
            max_depth = None # Default

            if model_type and "Random Forest" in model_type:
                n_estimators = st.slider("Number of Estimators (Random Forest)", 10, 300, 100, 10)
                max_depth_option = st.slider("Max Depth (Random Forest)", 1, 30, 10, 1)
                use_max_depth = st.checkbox("Limit Max Depth", value=True)
                max_depth = max_depth_option if use_max_depth else None


        # Submit button for the form
        submitted = st.form_submit_button("🚀 Fit Model & Evaluate")

    # --- Model Training & Results (only run if form submitted) ---
    if submitted:
        if not target_var:
            st.error("Please select a target variable.")
        elif not selected_features:
            st.error("Please select at least one feature.")
        elif task_type is None:
             st.error("Could not determine task type. Ensure target variable is selected.")
        elif not model_type:
             st.error("Please select a valid model for the detected task type.")
        else:
            st.header("3. Results")
            try:
                # --- 1. Data Preprocessing ---
                X = df[selected_features].copy() # Work on a copy
                y = df[target_var].copy()

                # Handle categorical features using one-hot encoding (simple approach)
                if selected_categorical_features:
                    X = pd.get_dummies(X, columns=selected_categorical_features, drop_first=True, dtype=int)

                # Get final feature names after potential dummy creation
                final_feature_names = X.columns.tolist()

                # Handle categorical target (encode if Classification)
                label_classes = None
                if task_type == "Classification":
                    # Store original labels if object/category before encoding
                    if not pd.api.types.is_numeric_dtype(y):
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        label_classes = le.classes_
                        st.info(f"Target variable encoded. Classes: {list(label_classes)}")
                    else:
                        # If target is numeric but treated as classification, get unique sorted values as classes
                        label_classes = np.sort(y.unique())


                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=(y if task_type=='Classification' else None) # Stratify for classification
                )
                st.write(f"Data Split: Training set has {X_train.shape[0]} samples, Test set has {X_test.shape[0]} samples.")

                # --- 2. Model Instantiation ---
                model = None
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                elif model_type == "Logistic Regression":
                    # Check again for binary classification before instantiating Logistic Regression
                    if len(np.unique(y)) != 2:
                         st.error("Logistic Regression currently implemented for binary classification only. Please choose Random Forest Classifier for multi-class problems.")
                         st.stop() # Stop execution here
                    model = LogisticRegression(random_state=42, max_iter=1000) # Increase max_iter for convergence
                elif model_type == "Random Forest Classifier":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

                # --- 3. Model Training ---
                model.fit(X_train, y_train)
                st.success(f"Model '{model_type}' trained successfully!")

                # --- 4. Prediction ---
                y_pred = model.predict(X_test)
                y_prob = None # Initialize y_prob
                if task_type == "Classification":
                    # Need probabilities for ROC curve (only for binary classification)
                    if hasattr(model, "predict_proba"):
                         # Check if binary classification for standard ROC
                         if len(model.classes_) == 2:
                             y_prob = model.predict_proba(X_test)[:, 1] # Probability of positive class
                         else:
                             st.info("ROC curve is typically shown for binary classification.")
                             # For multi-class, you might need different approaches (e.g., one-vs-rest) - skip for simplicity now
                    else:
                         st.warning(f"{model_type} does not support predict_proba. ROC curve cannot be generated.")


                # --- 5. Evaluation & Visualization ---
                st.subheader("Performance Metrics")
                results_col1, results_col2 = st.columns(2) # Columns for metrics and plots

                with results_col1:
                    if task_type == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                        st.metric("R-squared (R2)", f"{r2:.4f}")
                    elif task_type == "Classification":
                        acc = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", f"{acc:.4f}")
                        # Display Confusion Matrix Data
                        cm = confusion_matrix(y_test, y_pred)
                        # Use stored label_classes if available, otherwise model's classes_
                        display_labels = label_classes if label_classes is not None else (model.classes_ if hasattr(model, 'classes_') else None)
                        if display_labels is not None:
                            st.write("**Confusion Matrix:**")
                            cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
                            st.dataframe(cm_df)
                        else:
                             st.write("**Confusion Matrix (numeric labels):**")
                             st.dataframe(pd.DataFrame(cm))


                st.subheader("Visualizations")
                # Ensure plots have enough space
                plot_container = st.container()
                with plot_container:
                    plot_col1, plot_col2 = st.columns(2)

                    with plot_col1:
                        # Plot 1: Residuals (Regression) or Confusion Matrix (Classification)
                        plt.figure(figsize=(6, 4)) # Create figure explicitly
                        if task_type == "Regression":
                            st.write("**Residual Distribution**")
                            residuals = y_test - y_pred
                            sns.histplot(residuals, kde=True)
                            plt.title("Residual Distribution")
                            plt.xlabel("Residuals (Actual - Predicted)")
                            plt.tight_layout()
                            st.pyplot(plt.gcf()) # Pass current figure
                            plt.clf() # Clear figure for next plot
                        elif task_type == "Classification":
                            st.write("**Confusion Matrix Heatmap**")
                            cm = confusion_matrix(y_test, y_pred)
                            display_labels = label_classes if label_classes is not None else (model.classes_ if hasattr(model, 'classes_') else None)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=display_labels if display_labels is not None else 'auto',
                                        yticklabels=display_labels if display_labels is not None else 'auto')
                            plt.xlabel("Predicted Label")
                            plt.ylabel("True Label")
                            plt.title("Confusion Matrix")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()

                    with plot_col2:
                        # Plot 2: Feature Importance or ROC Curve (Classification)
                        plt.figure(figsize=(6, 4)) # Create figure explicitly
                        importance_df = None
                        if hasattr(model, 'feature_importances_'): # Random Forest
                            importances = model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': final_feature_names,
                                'Importance': importances
                            }).sort_values(by='Importance', ascending=False)

                        elif hasattr(model, 'coef_') and model_type != "Logistic Regression": # Linear Regression (avoid for multi-class Logistic)
                            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                            importances = abs(coefs)
                            importance_df = pd.DataFrame({
                                'Feature': final_feature_names,
                                'Importance (Absolute Coef)': importances # Label clarifies it's coef magnitude
                            }).sort_values(by='Importance (Absolute Coef)', ascending=False)

                        elif hasattr(model, 'coef_') and model_type == "Logistic Regression" and len(model.classes_) == 2: # Binary Logistic Regression
                            coefs = model.coef_[0] # Access the single set of coefficients
                            importances = abs(coefs)
                            importance_df = pd.DataFrame({
                                'Feature': final_feature_names,
                                'Importance (Absolute Coef)': importances
                            }).sort_values(by='Importance (Absolute Coef)', ascending=False)


                        if importance_df is not None:
                                st.write("**Feature Importance**")
                                # Show top N features for clarity
                                top_n = min(len(importance_df), 15)
                                sns.barplot(x=importance_df.columns[1], y='Feature', data=importance_df.head(top_n), palette="viridis")
                                plt.title(f"Top {top_n} Feature Importances")
                                plt.tight_layout()
                                st.pyplot(plt.gcf())
                                plt.clf()

                        elif task_type == "Classification" and y_prob is not None and len(np.unique(y_test)) == 2: # Ensure binary for ROC
                            st.write("**ROC Curve**")
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic (ROC) Curve')
                            plt.legend(loc="lower right")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()
                        elif task_type == "Classification" and len(np.unique(y_test)) != 2:
                             st.info("ROC curve is only shown for binary classification tasks.")
                        elif task_type == "Classification" and y_prob is None:
                             st.info("ROC curve could not be generated (predict_proba not available or not applicable).")
                        else: # Regression case where importance wasn't generated (shouldn't happen with current models)
                             st.info("Feature importance not available for this model configuration.")

            except KeyError as e:
                 st.error(f"KeyError during processing: {e}. This might happen if the selected features/target are no longer valid after dropping NA values or changing datasets. Please re-select features/target.")
                 st.exception(e) # Show traceback for debugging
            except ValueError as e:
                 st.error(f"ValueError: {e}. Check if data types are suitable for the selected model (e.g., non-numeric data in features for linear models without encoding, or trying Logistic Regression on multi-class target).")
                 st.exception(e)
            except Exception as e:
                 st.error(f"An unexpected error occurred during model training or evaluation: {e}")
                 st.exception(e) # Provides traceback for debugging

else:
    st.info("Please load a dataset using the options in the sidebar to get started.")
