from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Machine Learning Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from keras.preprocessing.sequence import TimeseriesGenerator

# Database Connection
import mysql.connector
from connection import connect_to_database, fetch_data_from_database

app = Flask(__name__)
CORS(app)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def make_predictions_and_update_df(model, current_batch, n_input, n_features, num_months, new_df, scaler, start_date, scaled_extended_train, selected_province):
    predictions = []

    current_batch = scaled_extended_train[-n_input:].reshape((1, n_input, n_features))

    for i in range(num_months):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(predictions)

    # Add predictions to the DataFrame
    pred_dates = pd.date_range(start=start_date, periods=num_months, freq='MS')
    pred_df = pd.DataFrame(index=pred_dates, columns=['Predictions'])
    pred_df['Predictions'] = true_predictions

    # Extract predictions for the selected province from extended_pred_df
    selected_province_predictions = pred_df[['Predictions']].rename(columns={'Predictions': selected_province})
    
    # Add 2024 predictions to the new_df DataFrame
    existing_dates = selected_province_predictions.index.intersection(new_df.index)

    # Remove the 'Predictions' column
    #selected_province_predictions = selected_province_predictions.drop(columns=['Predictions'])

    if not existing_dates.empty:
        # If some dates already exist, update only the missing dates
        new_dates = selected_province_predictions.index.difference(new_df.index)
        new_df = pd.concat([new_df, selected_province_predictions.loc[new_dates]], axis=0)
    else:
        # If no dates exist, concatenate the entire DataFrame
        new_df = pd.concat([new_df, selected_province_predictions], axis=0)

    # Drop duplicate rows based on the index (Date)
    new_df = new_df[~new_df.index.duplicated(keep='last')]

    return new_df, pred_df

def store_predictions_in_database(predictions_df, province):
    # Store predictions in the database
    conn = connect_to_database()
    cursor = conn.cursor()

    for date, prediction in zip(predictions_df.index, predictions_df['Predictions']):
        formatted_date = date.strftime('%Y-%m-%d')

        # Check if the record already exists for the given date and province
        check_query = f"SELECT * FROM predictionstb WHERE Date = '{formatted_date}' AND Province = '{province}';"
        cursor.execute(check_query)
        existing_record = cursor.fetchone()

        if existing_record:
            # Update the existing record if it already exists
            update_query = f"UPDATE predictionstb SET Predictions = {prediction} WHERE Date = '{formatted_date}' AND Province = '{province}';"
            cursor.execute(update_query)
        else:
            # Insert a new record if it doesn't exist
            insert_query = f"INSERT INTO predictionstb (Date, Province, Predictions) VALUES ('{formatted_date}', '{province}', {prediction});"
            cursor.execute(insert_query)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

def get_prediction_pivot(conn, province, preferredYear):
    # Fetch data from the predictionstb table for the selected province
    prediction_data = fetch_data_from_database(conn, f"SELECT * FROM predictionstb WHERE Province = '{province}';")

    # Convert the fetched data to a DataFrame
    prediction_df = pd.DataFrame(prediction_data, columns=['Date', 'Province', 'Predictions'])

    # Convert Date column to datetime format
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    # Filter the DataFrame to include only the data for the year X
    prediction_df = prediction_df[prediction_df['Date'].dt.year == preferredYear]

    # Convert the 'Date' column to string in the DataFrame
    prediction_df['Date'] = prediction_df['Date'].dt.strftime('%Y-%m-%d')

    # Pivot the DataFrame for the year X
    prediction_pivot = prediction_df.pivot(index='Date', columns='Province', values='Predictions')

    # Reset index to make 'Date' a regular column again
    prediction_pivot.reset_index(inplace=True)

    return prediction_pivot

# function to load data, preprocess
def load_data():
    # Connect to the database
    conn = connect_to_database()

    # SQL query to fetch data
    sql_query = "SELECT * FROM swinetb;"

    # Fetch data from the database
    data = fetch_data_from_database(conn, sql_query)

    # Convert the fetched data to a DataFrame
    df = pd.DataFrame(data, columns=['id','Date','PHILIPPINES','Cordillera','Abra','Apayao','Benguet','Ifugao','Kalinga','MountainProvince','REGIONI',
                                        'IlocosNorte','IlocosSur','LaUnion','Pangasinan','REGIONII','Batanes','Cagayan','Isabela','NuevaVizcaya','Quirino',
                                        'REGIONIII','Aurora','Bataan','Bulacan','NuevaEcija','Pampanga','Tarlac','Zambales','REGIONIVA','Batangas','Cavite','Laguna',
                                        'Quezon','Rizal','MIMAROPAREGION','Marinduque','OccidentalMindoro','OrientalMindoro','Palawan','Romblon','REGIONV','Albay',
                                        'CamarinesNorte','CamarinesSur','Catanduanes','Masbate','Sorsogon','REGIONVI','Aklan','Antique','Capiz','Guimaras','Iloilo',
                                        'NegrosOccidental','REGIONVII','Bohol','Cebu','NegrosOriental','Siquijor','REGIONVIII','Biliran','EasternSamar','Leyte',
                                        'NorthernSamar','Samar','SouthernLeyte','REGIONIX','ZamboangadelNorte','ZamboangadelSur','ZamboangaSibugay','ZamboangaCity',
                                        'REGIONX','Bukidnon','Camiguin','LanaodelNorte','MisamisOccidental','MisamisOriental','REGIONXI','DavaodeOro','DavaodelNorte',
                                        'DavaodelSur','DavaoOccidental','DavaoOriental','CityofDavao','REGIONXII','Cotabato','Sarangani','SouthCotabato','SultanKudarat',
                                        'REGIONXIII','AgusandelNorte','AgusandelSur','SurigaodelNorte','SurigaodelSur','MUSLIMMINDANAO','Basilan','Maguindanao','Sulu'])

    columns = df.columns[2:]

    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.drop(columns=['id'], inplace=True)  # Drop the 'id' column
    df.sort_index(inplace=True)

    # Convert non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns with more than 10% null values
    # df.drop(['National Capital Region (NCR)','Lanao del Sur', 'Tawi-tawi'], axis=1, inplace=True)
    # df.drop(['DavaoOccidental','Batanes', 'Sulu','CityofDavao', 'ZamboangaCity','Maguindanao', 'Basilan'], axis=1, inplace=True)
    # From 97 provinces to 90. 7 provinces had more than 10% null values and were dropped

    # Fill NaN values with the mean
    df.fillna(df.mean(), inplace=True)

    # Initialize ph_df with a default value
    selected_province = 'PHILIPPINES'
    ph_df = pd.DataFrame(df, columns=[selected_province])

    return ph_df, columns, df, conn

# Assume you have a function to train and save the LSTM model
def train_lstm_model(data, province):
    # Add your logic to preprocess data, train the LSTM model, and save it
    # For demonstration purposes, a simple LSTM model is trained
    n_input = 18
    n_features = 1

    train = data.iloc[:120]
    test = data.iloc[120:]

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    val_data = scaled_train[96:]

    generator_train = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    generator_val = TimeseriesGenerator(val_data, val_data, length=n_input, batch_size=1)

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer='l2'))

    optimizer = Adam(learning_rate=0.001, decay=1e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

    # model filename based on selection
    model_filename = f'model/lstm_model_{province.lower()}.h5'

    try:
        model = load_model(model_filename)
    except:
        model.fit(generator_train, epochs=100, validation_data=generator_val, callbacks=[early_stop])
        # Save the model with the selected province in the filename
        model.save(model_filename)
    
    return model, test, scaled_train, scaler, n_input, n_features

# Assume you have a route to train the model
@app.route('/train-model', methods=['GET'])
def train_model():
    selected_province = request.args.get('province', 'PHILIPPINES')
    ph_df, columns, df, conn = load_data()

    train_lstm_model(ph_df, selected_province)

    return f"Model training for {selected_province} completed successfully"

# Assume you have a route to get historical prices
@app.route('/api/historical-prices', methods=['GET'])
def get_historical_prices():
    selected_province = request.args.get('province', 'PHILIPPINES')

    # Load data for the selected province
    ph_df, columns, df, conn = load_data()

    ph_df = pd.DataFrame(df, columns=[selected_province])

    # Assuming "Date" is the date column
    historical_prices_df = ph_df.reset_index()  # Reset the index to include "Date"
    historical_prices_df['Date'] = pd.to_datetime(historical_prices_df['Date'])
    historical_prices_df['Month'] = historical_prices_df['Date'].dt.strftime('%b')
    historical_prices_df['Year'] = historical_prices_df['Date'].dt.year 
    historical_prices = historical_prices_df.to_dict(orient='records')

    return jsonify({"historicalPrices": historical_prices})

# Assume you have a route to get predicted prices
@app.route('/api/predicted-prices', methods=['GET'])
def get_predicted_prices():
    selected_province = request.args.get('province', 'PHILIPPINES')

    ph_df, columns, df, conn = load_data()

    ph_df = pd.DataFrame(df, columns=[selected_province])

    model, test, scaled_train, scaler, n_input, n_features = train_lstm_model(ph_df, selected_province)

    # Load the trained model
    # model = load_model(f'model/lstm_model_{province.lower()}.h5')

    # Add your logic to generate predicted prices
    # For demonstration purposes, using random data
    n_input = 18
    n_features = 1

    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)

    test['Predictions'] = true_predictions

     # Calculate accuracy metrics
    rmse = round(np.sqrt(mean_squared_error(test[selected_province], test['Predictions'])),2)
    mae = round(mean_absolute_error(test[selected_province], test['Predictions']),2)
    mape = round(mean_absolute_percentage_error(test[selected_province], test['Predictions']),2)
    r_squared = round(r2_score(test[selected_province], test['Predictions']),3)

    # Plot actual vs predicted values for 2023
    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test[selected_province], label='Actual Data', linestyle='--', color='blue')
    plt.plot(test.index, test['Predictions'], label='Predicted Data', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    predicted_plot_path = 'static/actual_vs_predicted.png'
    plt.savefig(predicted_plot_path)
    plt.close()

    # Prepare the table for comparison
    comparison_df = test
    
    comparison_df = comparison_df.reset_index()
    comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m-%d')

    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
    comparison_df['Month'] = comparison_df['Date'].dt.strftime('%b')
    comparison_df['Year'] = comparison_df['Date'].dt.year 

    # Extract the Date column and convert the DataFrame to a list of dictionaries
    comparison_data = comparison_df.round(2).reset_index().to_dict(orient='records')

    response_data = {
        "predictedPrices": comparison_data,
        "predictedPlot": predicted_plot_path,
        "accuracyMetrics": {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R_squared": r_squared
        }
    }

    return jsonify(response_data)

@app.route('/api/provinces', methods=['GET'])
def get_provinces():

    ph_df, columns, df, conn = load_data()
    provinces = df.columns.tolist()  # Get the list of provinces from the DataFrame
    return jsonify({"provinces": provinces})

@app.route('/api/predictions', methods=['GET', 'POST'])
def predictions_api():
    ph_df, columns, df, conn = load_data()

    # Get parameters from the request
    selected_province = request.args.get('province', 'PHILIPPINES')
    selected_year = request.args.get('year', '2024')

    predictions_df = get_prediction_pivot(conn, selected_province, int(selected_year))

    model, test, scaled_train, scaler, n_input, n_features = train_lstm_model(ph_df, selected_province)

    # Make predictions for the selected year
    num_months = 12  # Number of months
    new_df = pd.DataFrame(df, columns=[selected_province])
    scaled_train_data = scaler.transform(new_df)
    current_batch = scaled_train_data[-n_input:].reshape((1, n_input, n_features))

    updated_df, df_predictions = make_predictions_and_update_df(
        model, current_batch, n_input, n_features, num_months, new_df, scaler, f'{selected_year}-01-01', scaled_train_data,
        selected_province
    )

    store_predictions_in_database(df_predictions, selected_province)

    # Fetch predictions for the selected province and year
    predictions_df = get_prediction_pivot(conn, selected_province, int(selected_year))

    predictions_df = predictions_df.reset_index()

    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df['Month'] = predictions_df['Date'].dt.strftime('%b')
    predictions_df['Year'] = predictions_df['Date'].dt.year 

    predictions = predictions_df.to_dict(orient='records')

    # Prepare data for JSON response
    response_data = {
        'predictions': predictions
    }

    return jsonify(response_data)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8080)