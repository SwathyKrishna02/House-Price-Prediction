import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model (Ensure that 'rfModel.pkl' contains only the model, not encoders)
with open('rfModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define label encoders (these should match the ones used during model training)
label_encoders = {
    'Area Type': LabelEncoder(),
    'Area Locality': LabelEncoder(),
    'City': LabelEncoder(),
    'Furnishing Status': LabelEncoder(),
    'Tenant Preferred': LabelEncoder()
}

# Example: Fit the label encoders with the classes used during training.
# These should be the exact unique values used when training the model.
# Replace these with the actual classes used during your model training.

label_encoders['Area Type'].classes_ = np.array(['Super Area', 'Carpet Area', 'Built Area'])
label_encoders['Area Locality'].classes_ = np.array(['Bandel', 'Phool Bagan, Kankurgachi', 'Salt Lake City Sector 2',
       'BN Reddy Nagar', 'Godavari Homes, Quthbullapur', 'Manikonda, Hyderabad'])
label_encoders['City'].classes_ = np.array(['Kolkata', 'Mumbai', 'Bangalore', 'Delhi','Chennai','Hyderabad'])
label_encoders['Furnishing Status'].classes_ = np.array(['Unfurnished', 'Semi-Furnished', 'Furnished'])
label_encoders['Tenant Preferred'].classes_ = np.array(['Family', 'Bachelors','Bachelors/Family'])

# App title
st.title("House Rent Prediction App")

# User inputs
st.header("Enter the house details:")

# Dropdown for BHK
BHK_options = [1, 2, 3, 4, 5]
BHK = st.selectbox("BHK (Number of Bedrooms):", BHK_options)

# Dropdown for Size (in square feet)
Size_options = [i*100 for i in range(0, 101)]  # From 100 to 10000 square feet
Size = st.selectbox("Size of the house (in square feet):", Size_options)

# Dropdowns for Area Type, Locality, City, Furnishing Status, and Tenant Preferred
Area_Type = st.selectbox("Area Type:", label_encoders['Area Type'].classes_)
Area_Locality = st.selectbox("Area Locality:", label_encoders['Area Locality'].classes_)
City = st.selectbox("City:", label_encoders['City'].classes_)
Furnishing_Status = st.selectbox("Furnishing Status:", label_encoders['Furnishing Status'].classes_)
Tenant_Preferred = st.selectbox("Tenant Preferred:", label_encoders['Tenant Preferred'].classes_)

# Dropdown for number of Bathrooms
Bathroom_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Bathroom = st.selectbox("Number of Bathrooms:", Bathroom_options)

# Predict rent
if st.button("Predict Rent"):
    # Encode inputs
    Area_Type_encoded = label_encoders['Area Type'].transform([Area_Type])[0]
    Area_Locality_encoded = label_encoders['Area Locality'].transform([Area_Locality])[0]
    City_encoded = label_encoders['City'].transform([City])[0]
    Furnishing_Status_encoded = label_encoders['Furnishing Status'].transform([Furnishing_Status])[0]
    Tenant_Preferred_encoded = label_encoders['Tenant Preferred'].transform([Tenant_Preferred])[0]

    # Prepare features for prediction
    features = np.array([[BHK, Size, Area_Type_encoded, Area_Locality_encoded, City_encoded,
                          Furnishing_Status_encoded, Tenant_Preferred_encoded, Bathroom]])
    
    # Predict
    predicted_rent = model.predict(features)
    
    # Display result
    st.success(f"Predicted Monthly Rent: ₹{predicted_rent[0]:,.2f}")
