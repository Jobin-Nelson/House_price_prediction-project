import streamlit as st
import pandas as pd
import pickle, json

# loading all files
with open('./model/columns.json', 'r') as f:
    columns = json.load(f)['columns']
    locations = columns[3:]

with open('./model/house_price_model.pickle', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('../data/Bengaluru_House_Data.csv')
df['rooms'] = df['size'].map(lambda x: int(x.split(' ')[0]), na_action='ignore')
df.drop(['area_type', 'society', 'balcony', 'availability', 'size'], axis=1, inplace=True)

def predict_price(location, sqft, bath, room):
    if sqft.isnumeric():
        sqft = float(sqft)
        x = [0]*len(columns)

        if location in locations:
            loc_ind = columns.index(location.lower())
        else:
            loc_ind = -1
        
        x[0], x[1], x[2], x[loc_ind] = sqft, bath, room, 1
        pred = pd.DataFrame([x], columns=columns)
        return round(model.predict(pred)[0],2)
    else:
        return 'Quit it, Give me a number'

# website structure
header = st.container()
user_specified = st.container()
dataset = st.container()

with header:
    st.title('Bangalore House Price Prediction')
    st.write('Predicts the price of house based on user inputs')
    st.write('---')

with user_specified:
    user_inputs, price = st.columns(2)

    user_inputs.header('User inputs')
    location = user_inputs.text_input('Name of location', 'Whitefield')
    sqft = user_inputs.text_input('Area in square feet', 1200)
    bath = user_inputs.slider('No. of Bathrooms', min_value=1, max_value=12, value=3, step=1)
    room = user_inputs.slider('No. of Rooms', min_value=1, max_value=12, value=4, step=1)

    estimated_price = predict_price(location, sqft, bath, room)

    price.header('Estimated Price')
    price.subheader(estimated_price)

with dataset:
    st.markdown("## [Dataset](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)")
    st.write("Take a peek at the dataset on which the model was trained")
    st.write(df)