import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from io import StringIO
import base64
import streamlit as st

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{base64_str}"  # Ensure correct MIME type

# Use raw string or forward slashes to prevent errors
image_path = r"C:\Users\Meyyarivu\OneDrive\Desktop\streamlit project\bull-bear-trading-3840x2160-13849.png"
# Get base64 encoded image
base64_image = get_base64_image(image_path)

# Print first 100 characters to verify it's correctly generated
print(base64_image[:100])

# Correctly insert the base64 string into the CSS using an f-string
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{base64_image}");
        background-size: cover;
        background-position: center;
        filter: brightness(90%);
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Title for the app
st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color:rgb(0, 255, 76); 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        border: 3px solidrgb(60, 255, 0);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        Stock Price Prediction App
    </h1>
    """,
    unsafe_allow_html=True,
)
# Upload CSV file

# Load the dataset
option = st.sidebar.selectbox(
    "Choose a page:",
    ["Tesla", "Reliance", "AAPL"]
)
st.markdown( "Watch This Video for More Insights")
st.video("https://youtu.be/bqPSFw1eiNc?si=7u-28bHqgktABSaH")




def preprocess_data(df):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ["int64", "float64"]:
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
    return df
    


 ########################################################################################################################################
        

if option == "Tesla":
    
    st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color:rgb(0, 255, 64); 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        TESLA Stock Price Prediction
    </h1>
    """,
    unsafe_allow_html=True,
)  
     
     
    data = pd.read_csv(r"C:\Users\Meyyarivu\OneDrive\Desktop\streamlit project\TSLA.csv")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('Home'):
            st.session_state.page = 'Home'
    with col2:
        if st.button('Data Preprocessing'):
            st.session_state.page = 'Data Preprocessing'
    with col3:
        if st.button('Model Building'):
            st.session_state.page = 'Model Building'
    with col4:
        if st.button('Visualization'):
            st.session_state.page = 'Visualization'
    with col5:
        if st.button('Prediction'):
            st.session_state.page = 'Prediction'


    col1, col2, col3 ,col4= st.columns(4)
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    regressor = LinearRegression()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})
    
    
    
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if st.session_state.page == "Home":
        st.markdown(f"<h2 style='color:#ff0078d7; font-weight:bold;'>Welcome to the Stock Price Prediction App</h2>"
    
          
         , unsafe_allow_html=True)

  
      
      

    elif st.session_state.page == "Data Preprocessing":
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Overview:</h4>", unsafe_allow_html=True)
            st.write(data.head())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (Before Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            
            data = preprocess_data(data)

            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (After Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Information:</h4>", unsafe_allow_html=True)
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Description:</h4>", unsafe_allow_html=True)
            st.write(data.describe())
    
    
    # Train-test split
    elif st.session_state.page == "Visualization":
        graph = data1.head(20)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>LINE PLOT</h4>", unsafe_allow_html=True)

        columns = st.multiselect("Select columns to visualize", data.columns.tolist(),
                                 default=["High", "Low", "Open", "Close"])
        if columns:
            st.line_chart(data[columns])
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>BAR GRAPH (Actual Vs Predicted)</h4>", unsafe_allow_html=True)

        st.bar_chart(graph)
    
    elif st.session_state.page == "Model Building":
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Coefficients </h4>", unsafe_allow_html=True)
        st.write(regressor.coef_)

        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Actual Vs Predicted </h4>", unsafe_allow_html=True)
        st.write(data1.head(20))
        mae = metrics.mean_absolute_error(y_test, predicted)
        mse = metrics.mean_squared_error(y_test, predicted)
        rmse = math.sqrt(mse)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Intercept: {regressor.intercept_}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Absolute Error: {mae}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Squared Error: {mse}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Root Mean Squared Error: {rmse}</h4>", unsafe_allow_html=True)
    
    
    elif st.session_state.page == "Prediction":
    # New data prediction
        st.write("Predict New Data:")
        high = st.number_input("High Price:")
        low = st.number_input("Low Price:")
        open_price = st.number_input("Open Price:")
        volume = st.number_input("Volume:")

        if st.button("Predict"):
            new_data = np.array([[high, low, open_price, volume]])
            predicted_price = regressor.predict(new_data)
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Predicted Close Price  : {predicted_price[0]}</h4>", unsafe_allow_html=True)


            
###########################################################################################################################################
if option == "Reliance":
    
    st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color:rgb(8, 237, 54); 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        TESLA Stock Price Prediction
    </h1>
    """,
    unsafe_allow_html=True,
)  
     
     
    data = pd.read_csv(r"C:\Users\Meyyarivu\OneDrive\Desktop\streamlit project\RELIANCE_CLEANED.csv")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('Home'):
            st.session_state.page = 'Home'
    with col2:
        if st.button('Data Preprocessing'):
            st.session_state.page = 'Data Preprocessing'
    with col3:
        if st.button('Model Building'):
            st.session_state.page = 'Model Building'
    with col4:
        if st.button('Visualization'):
            st.session_state.page = 'Visualization'
    with col5:
        if st.button('Prediction'):
            st.session_state.page = 'Prediction'


    col1, col2, col3 ,col4= st.columns(4)
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    regressor = LinearRegression()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})
    
    
    
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if st.session_state.page == "Home":
        st.markdown(f"<h2 style='color:#ff0078d7; font-weight:bold;'>Welcome to the Stock Price Prediction App</h2>", unsafe_allow_html=True)

        
    elif st.session_state.page == "Data Preprocessing":
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Overview:</h4>", unsafe_allow_html=True)
            st.write(data.head())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (Before Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            
            data = preprocess_data(data)

            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (After Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Information:</h4>", unsafe_allow_html=True)
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Description:</h4>", unsafe_allow_html=True)
            st.write(data.describe())
    
    
    # Train-test split
    elif st.session_state.page == "Visualization":
        graph = data1.head(20)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>LINE PLOT</h4>", unsafe_allow_html=True)

        columns = st.multiselect("Select columns to visualize", data.columns.tolist(),
                                 default=["High", "Low", "Open", "Close"])
        if columns:
            st.line_chart(data[columns])
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>BAR GRAPH (Actual Vs Predicted)</h4>", unsafe_allow_html=True)

        st.bar_chart(graph)
    
    elif st.session_state.page == "Model Building":
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Coefficients </h4>", unsafe_allow_html=True)
        st.write(regressor.coef_)

        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Actual Vs Predicted </h4>", unsafe_allow_html=True)
        st.write(data1.head(20))
        mae = metrics.mean_absolute_error(y_test, predicted)
        mse = metrics.mean_squared_error(y_test, predicted)
        rmse = math.sqrt(mse)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Intercept: {regressor.intercept_}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Absolute Error: {mae}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Squared Error: {mse}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Root Mean Squared Error: {rmse}</h4>", unsafe_allow_html=True)
    
    
    elif st.session_state.page == "Prediction":
    # New data prediction
        st.write("Predict New Data:")
        high = st.number_input("High Price:")
        low = st.number_input("Low Price:")
        open_price = st.number_input("Open Price:")
        volume = st.number_input("Volume:")

        if st.button("Predict"):
            new_data = np.array([[high, low, open_price, volume]])
            predicted_price = regressor.predict(new_data)
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Predicted Close Price  : {predicted_price[0]}</h4>", unsafe_allow_html=True)


            
###########################################################################################################################################
if option == "AAPL":
    
    st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color:rgb(0, 255, 64); 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        TESLA Stock Price Prediction
    </h1>
    """,
    unsafe_allow_html=True,
)  
     
     
    data = pd.read_csv(r"C:\Users\Meyyarivu\OneDrive\Desktop\streamlit project\AAPL.csv")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('Home'):
            st.session_state.page = 'Home'
    with col2:
        if st.button('Data Preprocessing'):
            st.session_state.page = 'Data Preprocessing'
    with col3:
        if st.button('Model Building'):
            st.session_state.page = 'Model Building'
    with col4:
        if st.button('Visualization'):
            st.session_state.page = 'Visualization'
    with col5:
        if st.button('Prediction'):
            st.session_state.page = 'Prediction'


    col1, col2, col3 ,col4= st.columns(4)
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    regressor = LinearRegression()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})
    
    
    
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if st.session_state.page == "Home":
        st.markdown(f"<h2 style='color:#ff0078d7; font-weight:bold;'>Welcome to the Stock Price Prediction App</h2>", unsafe_allow_html=True)

        

      

    elif st.session_state.page == "Data Preprocessing":
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Overview:</h4>", unsafe_allow_html=True)
            st.write(data.head())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (Before Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            
            data = preprocess_data(data)

            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (After Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Information:</h4>", unsafe_allow_html=True)
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Description:</h4>", unsafe_allow_html=True)
            st.write(data.describe())
    
    
    # Train-test split
    elif st.session_state.page == "Visualization":
        graph = data1.head(20)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>LINE PLOT</h4>", unsafe_allow_html=True)

        columns = st.multiselect("Select columns to visualize", data.columns.tolist(),
                                 default=["High", "Low", "Open", "Close"])
        if columns:
            st.line_chart(data[columns])
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>BAR GRAPH (Actual Vs Predicted)</h4>", unsafe_allow_html=True)

        st.bar_chart(graph)
    
    elif st.session_state.page == "Model Building":
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Coefficients </h4>", unsafe_allow_html=True)
        st.write(regressor.coef_)

        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Actual Vs Predicted </h4>", unsafe_allow_html=True)
        st.write(data1.head(20))
        mae = metrics.mean_absolute_error(y_test, predicted)
        mse = metrics.mean_squared_error(y_test, predicted)
        rmse = math.sqrt(mse)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Intercept: {regressor.intercept_}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Absolute Error: {mae}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Squared Error: {mse}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Root Mean Squared Error: {rmse}</h4>", unsafe_allow_html=True)
    
    
    elif st.session_state.page == "Prediction":
    # New data prediction
        st.write("Predict New Data:")
        high = st.number_input("High Price:")
        low = st.number_input("Low Price:")
        open_price = st.number_input("Open Price:")
        volume = st.number_input("Volume:")

        if st.button("Predict"):
            new_data = np.array([[high, low, open_price, volume]])
            predicted_price = regressor.predict(new_data)
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Predicted Close Price  : {predicted_price[0]}</h4>", unsafe_allow_html=True)


            
###########################################################################################################################################
