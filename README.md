# Sales Prediction App
Welcome to the Sales Prediction App! This application allows you to predict sales based on various input features. It uses a machine learning model that has been trained on historical sales data.

## Getting Started
To use this app and make predictions, follow the steps below:

## 1. Clone the Repository
First, clone this repository to your local machine. You can do this by running the following command in your terminal or command prompt:


git clone <https://github.com/shimwagape/streamlit.git>
## 2. Navigate to the Project Directory
Change your working directory to the project folder:


cd <streamlit>
## 3. Install Required Dependencies
Make sure you have the required Python packages installed. You can install them using pip by running:


pip install streamlit numpy pillow xgboost
## 4. Run the App
You can run the Sales Prediction App by executing the provided Python script:

streamlit run app.py
## 5. Use the App
Once the app is running, you can interact with it through your web browser. It provides a user interface that allows you to input various sales-related data and get predictions.

## Input Features
The following are the input features you can adjust to make predictions:

Store Number: Use a slider to select the store number.

Products Family: Choose the product family from a dropdown list.

Discount Amount on Promotion: Input the discount amount on promotion.

State: Select the state from a dropdown list.

Store Type: Choose the store type.

Cluster: Input a cluster number.

Month: Use a slider to select the month.

Day: Use a slider to select the day.

DCOILWTICO: Use a slider to input the DCOILWTICO value.

### Making Predictions
After inputting the relevant data, click the "Predict" button.
Prediction Output
The predicted sales value will be displayed on the app.
Model Information
The app uses a machine learning model that has been trained to predict sales. The model components are loaded from the export directory during app startup. The model was trained on historical sales data.

For any questions or issues, feel free to reach out to the project maintainers. Thank you for using the Sales Prediction App!
## Author
SHIMWA Agape Valentin 
### email
savalentin75@gmail.com
