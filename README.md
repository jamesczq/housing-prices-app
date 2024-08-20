# Coding Exercise - California Housing Prices Prediction 

This project shows the main process of designing, implementing, and deploying a machine learning model that predicts housing prices. Notice that the focus is more on software architecture and deployment parts than on the machine learning model development part.

## Data Preprocessing

After getting the dataset [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data), we first did some data analysis, mainly, (1) identifying what the features are and what the target variables is, and (2) inspecting data to see if there were noticeable problems, e.g., missing-value problems, large presence of outliers. Fortunately, the data seems already clean. These were done through data visualization, correlation anlysis, etc. Details can be found in Section 1 of the notebook `notebooks/from-data-to-model.ipynb`.

### Feature Engineering

For simplicity, and acting out of firm belief in "Principle of Least Action", we defined the feature engineering to take the following simple actions:
* For categorical features, we did one-hot encoding.
* For numerical features, we imputed missing values with median and scaled the values with the standard scaling (i.e., the transformed feature values are with $\mu = 0, \sigma = 1$).

The details can be found in `src/preprocessing.py`.

## Machine Learning Model Creation

### Which Regression Algorithm to Use?
For simplicity, we went for classical machine learning models which training and inferencing can be done fast, compared with deep learning models. Further, we looked at two prototypical classical regression models: linear regression (with regularization, e.g., Lasso) and random forest regression which is non-linear.

We first did a few experiments: training regression Linear Regression models and Random Forest Regression models, and comparing their prediction performance. We observed from cross-validation estimates of mean absolute error (MAE) and test MAE that Random Forest Regressor seemed to outperform Linear Regressor. Thus, we focused on building a `Random Forest` regression model. The details of these experiments can be found in Section 4 of `notebooks/from-data-to-model.ipynb`.

### Model Training
We encapsulated the model in a pipeline. The steps are as follows:

1. Data Splitting: The dataset is split into training and testing datasets for model evaluation. 

2. Feature Engineering: 
  * The numerical features are scaled using Standard Scaler to ensure that all features have comparable scale, which improves the performance of the models. 
  * The categorical features are encoded using a one-hot (aka ‘one-of-K’) encoding scheme. This creates a binary column for each category. Thus all features are turned into numeric values.

3. Model Training: A Random Forest Regression model was trained on the training data.

4. Model Evaluation: The performance of the models were evaluated on the testing data using Mean Absolute Error (MAE).

The details of above can be found in `src/preprocessing.py` and `src/preprocessing.py`.

## Model Serving

We served the prediction/inference with REST API, which can accept input features in JSON format (*acceptable input features are defined in `src/schema.py`) and return the predicted housing price. 

The details can be found in `app.py`.

## Deployment

The application was containerized using Docker. 

To run the FastAPI server, follow these steps:

1. Clone the [housing-prices-app](https://github.com/jamesczq/housing-prices-app) repository
2. Navigate to the root of the repository
3. Build the Docker image: `docker build -t house-price-api-image .`
4. Run the Docker container: `docker run --name api_container -p 8000:8000 house-price-api-image`
5. The FastAPI server will be accessible at `http://localhost:8000`

## Testing/API Reference

You can interact with the API using the Swagger UI documentation. The UI documentation is automatically generated and can be accessed from the `/docs` endpoint.

### Health check

```
  GET /
```

#### Response

```json
{
  "response": "Ready!"
}
```

### Endpoint: /model-info

```
  GET /model-info
```

#### Response

```json
{
    "performance": "Mean Absolute Error (MAE) estimated from cross-validation: 55912.65 +/- 9802.62",
    "model-info": "Pipeline(steps=[('Preprocessing',\n ColumnTransformer(transformers=[('Numeric Feature Transform',\n Pipeline(steps=[('Imputer',\n SimpleImputer(strategy='median')),\n ('Std_Scaler',\n StandardScaler())]),\n ['longitude', 'latitude',\n 'housing_median_age',\n 'total_rooms',\n 'total_bedrooms',\n 'population', 'households',\n 'median_income']),\n ('Categorical Feature '\n 'Transform',\n Pipeline(steps=[('One-hot',\n OneHotEncoder(handle_unknown='ignore'))]),\n ['ocean_proximity'])])),\n ('Regression Model Random Forest',\n RandomForestRegressor(n_estimators=30))])"
}
```

### Endpoint: /predict

```
  POST /predict
```

This POST method expects a JSON object with the property details to predict its price. The JSON body should follow the schema below:

```json
{
    "longitude": -122.64,
    "latitude": 38.24,
    "housing_median_age": 40.0,
    "total_rooms": 1974.0,
    "total_bedrooms": 410.0,
    "population": 1039.0,
    "households": 398.0,
    "ocean_proximity": "<1H OCEAN"
}
```

#### Response:

A successful request to the API will return a JSON object containing the predicted property price:

```json
{
    "prediction": 215256,
    "status_code": 200
}
```
