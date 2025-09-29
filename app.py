# Streamlit app to load and use the saved Decision Tree model
import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import base64
import os

# Set page config for better appearance
st.set_page_config(page_title='Detecting Phishing', page_icon=':shield:', layout='centered')

# Add a custom header with style
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: #fff; text-align: center; font-size: 2.5rem; margin: 0;'>🛡️ Detecting Phishing</h1>
        <p style='color: #e0e0e0; text-align: center; font-size: 1.2rem;'>Upload your data and let our AI model detect phishing attempts!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Optional: Add a simple icon or illustration (SVG)
st.markdown(
    """
    <div style='text-align:center;'>
        <svg width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" fill="#2a5298"/>
        <path d="M12 6v6l4 2" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the data (for feature names and example)
df = pd.read_csv('phishingtest2.csv')
X = df.drop(columns=['label'])  # Features
feature_labels =   ['Number of dots in the URL', 'Number of hyphens in the URL', 'Number of slashes in the URL', 
                   'Number of equal signs in the URL', 'Number of "&" symbols in the URL', 'Number of percent signs in the URL', 
                   'Length of the URL', 'Number of dots in the domain', 'Number of vowels in the domain', 'Length of the domain', 
                   'Number of slashes in the directory', 'Number of percent signs in the directory', 'Length of the directory', 
                   'Number of percent signs in the file', 'Length of the file', 'Number of dots in the parameters', 
                   'Number of equal signs in the parameters', 'Number of "&" symbols in the parameters', 
                   'Number of percent signs in the parameters', 'Length of the parameters', 'Number of parameters', 'Time response', 
                   'ASN IP', 'Time domain activation', 'Time domain expiration', 'Number of IP addresses resolved', 
                   'Number of name servers', 'Number of MX servers', 'TTL hostname', 'Phishing indicator']      
feature_names = list(X.columns)

# Load the trained model
with open('C:/Users/nebiy/OneDrive/Documents/phishing streamlit/decision_tree_model.pkl', 'rb') as f:
    model_dt = pickle.load(f)
with open('C:/Users/nebiy/OneDrive/Documents/phishing streamlit/neural_network_model.pkl', 'rb') as f:
    model_nn = pickle.load(f)
with open('C:/Users/nebiy/OneDrive/Documents/phishing streamlit/log_reg_model.pkl', 'rb') as f:
    model_lr = pickle.load(f)    


# Add sidebar info
st.sidebar.title('About')
st.sidebar.info('This app utilizes Decision Tree, Neural Network, and Logistic Regression models to predict phishing websites.')
# Add man fishing JPG image to sidebar
image_path = os.path.join(os.getcwd(), 'man_fishing.jpg')
if os.path.exists(image_path):
    st.sidebar.image(image_path, use_container_width=True)
else:
    st.sidebar.warning('man_fishing.jpg not found in the current directory.')


st.write('Please upload your dataset below.')
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    # Ensure only the columns used for training are present
    input_df = input_df[X.columns]
    st.write('### Input Data:')
    st.dataframe(input_df)

    # Model selection buttons
    col1, col2, col3, col4 = st.columns(4)
    predict_dt = col1.button('Predict with Decision Tree')
    predict_mlp = col2.button('Predict with MLP')
    predict_lr = col3.button('Predict with Logistic Regression')
    # --- Essential tweak: persist classification report view using st.session_state ---
    # Initialize session state for report view if not present
    if 'show_report' not in st.session_state:
        st.session_state['show_report'] = False
    # When the user clicks 'Generate Classification Report', set the flag to True
    if col4.button('Generate Classification Report'):
        st.session_state['show_report'] = True

    # --- Essential tweak: use radio for model selection in report view ---
    if st.session_state['show_report']:
        st.subheader("Model Evaluation (on entire dataset):")
        # Use radio button to select which model to evaluate
        model_choice = st.radio('Select model for evaluation:', ['Decision Tree', 'Neural Network', 'Logistic Regression'])

        from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        df_eval = pd.read_csv('Phishing_clean.csv')
        y_eval = df_eval['label']
        X_eval = df_eval.drop(columns=['label'])

        # Evaluate the selected model
        if model_choice == 'Decision Tree':
            y_pred = model_dt.predict(X_eval)
        elif model_choice == 'Neural Network':
            y_pred = model_nn.predict(X_eval)
        elif model_choice == 'Logistic Regression':
            y_pred = model_lr.predict(X_eval)
        else:
            y_pred = None

        # Display metrics if prediction is available
        if y_pred is not None:
            st.write(f"Accuracy: {accuracy_score(y_eval, y_pred):.4f}")
            st.write(f"Precision: {precision_score(y_eval, y_pred):.4f}")
            st.write(f"Recall: {recall_score(y_eval, y_pred):.4f}")
            st.write(f"F1 Score: {f1_score(y_eval, y_pred):.4f}")
            st.subheader("Classification Report:")
            report = classification_report(y_eval, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            st.subheader("Confusion Matrix:")
            cm = confusion_matrix(y_eval, y_pred)
            st.write(cm)
        # --- Essential tweak: add button to return to main menu ---
        if st.button('Back to Main Menu'):
            st.session_state['show_report'] = False

    # Make prediction

    if predict_dt:
        predictions = model_dt.predict(input_df)
        st.success('Decision Tree Prediction complete!')
        st.write('### Predictions:')
        #st.write(predictions)
        if predictions == 1:
            st.write("<span style='color:red'>Likely from a Phishing Website</span>", unsafe_allow_html=True)
        else:
            st.write("<span style='color:green'>Likely from a Legitimate Website</span>", unsafe_allow_html=True)
    if predict_mlp:
        #st.warning('No model exists right now for this one.')
        predictions = model_nn.predict(input_df)
        st.success('Neural Network Prediction complete!')
        st.write('### Predictions:')
        #st.write(predictions)
        if predictions == 1:
            st.write("<span style='color:red'>Likely from a Phishing Website</span>", unsafe_allow_html=True)
        else:
            st.write("<span style='color:green'>Likely from a Legitimate Website</span>", unsafe_allow_html=True)
                
    if predict_lr:
        #st.warning('No model exists right now for this one.')
        predictions = model_lr.predict(input_df)
        st.success('Logistic Regression Prediction complete!')
        st.write('### Predictions:')
        #st.write(predictions)
        if predictions == 1:
            st.write("<span style='color:red'>Likely from a Phishing Website</span>", unsafe_allow_html=True)
        else:
            st.write("<span style='color:green'>Likely from a Legitimate Website</span>", unsafe_allow_html=True)