import streamlit as st
import pandas as pd
import joblib

# load model
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
model = joblib.load('model.pkl')

required_columns = [
    'Age', 'Gender', 'Education Level', 'Income', 'Credit Score', 'Loan Amount', 
    'Loan Purpose', 'Employment Status', 'Years at Current Job', 'Payment History',
    'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults'
]

def sync_slider_input_values(key):
    st.session_state[f"{key}_slider"] = st.session_state[f"{key}_input"]

def sidebar_slider_with_input(label, min_value, max_value, default_value):
    
    # position the slider and number input side by side
    col1, col2 = st.columns([3, 1])
    
    # Create the slider in the first column
    slider_value = col1.slider(label, min_value, max_value, default_value,key=f"{label}_slider")
    
    # Create the number input in the second column without a label
    input_value = col2.number_input("", min_value, max_value, slider_value,key=f"{label}_input",on_change=sync_slider_input_values,
                                    args=(label,))
    
    return slider_value


st.write("""
# Financial Risk Prediction
""")
st.caption("Machine Learning for Developers (CAI2C08) Project by Koh Keira (2302442E)")
st.write("Ever wondered how different factors influence financial risk? :thinking_face: Let's explore the insights!")


st.sidebar.header('User Input Parameters')

def user_input_features():

    with st.sidebar.expander("Upload Data"):
        uploaded_file = st.file_uploader("Upload only a sample with column headers",type=['csv'])
    
    with st.sidebar.expander("Input Data Manually"):
        age = sidebar_slider_with_input('Age', 18, 69,33)
        gender = st.selectbox('Gender', ("Female", "Male", "Non-binary"),2)
        education = st.selectbox('Education Level', ("High School", "Master's", "Bachelor's", "PhD"),3)
        income = sidebar_slider_with_input('Income', 20005,119997,90683)
        credit_score = sidebar_slider_with_input('Credit Score',600,799,744)
        loan_amount = sidebar_slider_with_input('Loan Amount',5000,49998,28885)
        loan_purpose = st.selectbox('Loan Purpose', ("Business", "Auto", "Home", "Personal"),1)
        employment = st.selectbox('Employment Status', ("Unemployed", "Employed", "Self-employed"),0)
        job_years = sidebar_slider_with_input('Years at Current Job',0,19,0)
        payment_history = st.selectbox('Payment History', ("Poor", "Fair", "Good", "Excellent"),0)
        debt_to_icome = sidebar_slider_with_input('Debt-to-Income Ratio',0.100004,0.599970,0.467426)
        assets_value = sidebar_slider_with_input('Assets Value',20055,299999,45037)
        dependent_no = st.selectbox('Number of Dependents', (0, 1, 2, 3, 4),2)
        previous_defaults = st.selectbox('Previous Defaults',(0, 1, 2, 3, 4),4)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
         # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Error: The following required columns are missing: {', '.join(missing_columns)}")
            return None, uploaded_file
            
    else:
        data = {'Age': age,
                'Gender': gender,
                'Education Level': education,
                'Income': income,
                'Credit Score': credit_score,
                'Loan Amount': loan_amount,
                'Loan Purpose': loan_purpose,
                'Employment Status': employment,
                'Years at Current Job': job_years,
                'Payment History': payment_history,
                'Debt-to-Income Ratio': debt_to_icome,
                'Assets Value': assets_value,
                'Number of Dependents': dependent_no,
                'Previous Defaults': previous_defaults,
               }
    
    features = pd.DataFrame(data, index=[0])
    return features, uploaded_file

df,uploaded_file = user_input_features()

if df is not None:
    st.subheader('User Input parameters',divider=True)
    
    
    # Split the DataFrame into two parts
    midpoint = len(df.columns)//2
    df_part1 = df.iloc[:, :midpoint]  
    df_part2 = df.iloc[:, midpoint:]
    
    # Display inputs
    st.dataframe(df_part1, use_container_width=True, hide_index=True)
    st.dataframe(df_part2, use_container_width=True, hide_index=True)
    
    if uploaded_file is not None:
        st.info("Clear file to input data manually.")    
    
    encoded_df = df.copy()
    
    ## Ordinal Encoding
    ordinal_cols_categories = {
        "Education Level":{"High School":0,"Bachelor's":1,"Master's":2,"PhD":3},
        "Payment History":{"Poor":0,"Fair":1,"Good":2,"Excellent":3}
    }
    for col, cat in ordinal_cols_categories.items():
        encoded_df[col] = encoded_df[col].map(cat)
    
    ## One Hot Encoding
    categorical_cols = ['Gender','Loan Purpose','Employment Status']
    
    one_hot_encoded = one_hot_encoder.transform(encoded_df[categorical_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    
    # Concatenate the one-hot encoded dataframe with the original dataframe
    encoded_df = pd.concat([encoded_df, one_hot_df], axis=1)
    
    # Drop the original categorical columns
    encoded_df = encoded_df.drop(categorical_cols, axis=1)
    
    prediction = model.predict(encoded_df)

    # Show message conditionally based on prediction
    if(prediction=="Low"):
        st.subheader('Prediction',divider='green')
        st.markdown('''
        :green[It seems like the risk level is low! Great news!] :partying_face:''')
    else:
        st.subheader('Prediction',divider='red')
        st.markdown('''
        :red[Oh no! It seems like the risk level is high!] :rotating_light:''')
        
    st.write(prediction)


