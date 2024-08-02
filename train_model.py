import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
import joblib

# Load the dataset
df = pd.read_csv('rainfall.csv')

# Data Preparation
df.drop(['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'], axis=1, inplace=True)

# Ensure no leading/trailing whitespaces in label values
df['SUBDIVISION'] = df['SUBDIVISION'].str.strip()

# Encode categorical features
label_encoder = LabelEncoder()
df['SUBDIVISION'] = label_encoder.fit_transform(df['SUBDIVISION'])

# Define the features and target columns
features = ['SUBDIVISION', 'YEAR']
target_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[target_columns] = imputer.fit_transform(df[target_columns])

# Scale the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_columns], test_size=0.2, random_state=42)

# Define a base model, an advanced model, and Random Forest for stacking
base_model = RidgeCV()
advanced_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
random_forest = RandomForestRegressor(random_state=42)

# Define the parameter grid for tuning
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Model Training for each month using stacking
models = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for month in target_columns:
    # Randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(estimator=advanced_model, param_distributions=param_dist, n_iter=50, cv=kf, 
                                       scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train[month])

    # Retrieve the best model from the search
    best_advanced_model = random_search.best_estimator_

    # Stacking models
    stacking_model = StackingRegressor(
        estimators=[('ridge', base_model), ('xgb', best_advanced_model), ('rf', random_forest)],
        final_estimator=RidgeCV()
    )
    stacking_model.fit(X_train, y_train[month])
    models[month] = stacking_model

    # Save the model
    joblib.dump(stacking_model, f'model_{month}.pkl')

# Save the label encoder, scaler, and imputer
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
