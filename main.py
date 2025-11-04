#triggering ci/cd in actions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers

def load_data(filepath):
    """Loads the hotel booking data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(hotel_df):
    """Preprocesses the hotel booking data."""
    X = hotel_df.copy()
    y = X.pop('is_canceled')

    X['arrival_date_month'] = \
        X['arrival_date_month'].map(
            {'January':1, 'February': 2, 'March':3,
             'April':4, 'May':5, 'June':6, 'July':7,
             'August':8, 'September':9, 'October':10,
             'November':11, 'December':12}
        )

    features_num = [
        "lead_time", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "required_car_parking_spaces",
        "total_of_special_requests", "adr",
    ]
    features_cat = [
        "hotel", "arrival_date_month", "meal",
        "market_segment", "distribution_channel",
        "reserved_room_type", "deposit_type", "customer_type",
    ]

    transformer_num = make_pipeline(
        SimpleImputer(strategy="constant"),
        StandardScaler(),
    )
    transformer_cat = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown='ignore'),
    )

    preprocessor = make_column_transformer(
        (transformer_num, features_num),
        (transformer_cat, features_cat),
    )

    X = preprocessor.fit_transform(X)

    return X, y

def split_data(X, y):
    """Splits the data into training, validation, and test sets."""
    X_train_full, X_valid, y_train_full, y_valid = train_test_split(
        X, y, stratify=y, train_size=0.75
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, stratify=y_train_full, train_size=0.75
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def build_model(input_shape):
    """Builds the neural network model."""
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile_model(model):
    """Compiles the neural network model."""
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_valid, y_valid):
    """Trains the neural network model."""
    early_stopping = keras.callbacks.EarlyStopping(
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=200,
        callbacks=[early_stopping],
        verbose=0 # Set to 1 or 2 for progress output
    )
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on the test data."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = 'hotel.csv' # Update this path if necessary

    # Load data
    hotel_df = load_data(dataset_path)

    # Preprocess data
    X, y = preprocess_data(hotel_df)

    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    # Define input shape
    input_shape = [X_train.shape[1]]

    # Build, compile, and train the model
    model = build_model(input_shape)
    model = compile_model(model)
    history = train_model(model, X_train, y_train, X_valid, y_valid)

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # You can add code here to save the trained model,

    # make predictions, or visualize results if needed.
