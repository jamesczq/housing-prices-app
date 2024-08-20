# CA Housing Prices

This project is focused on creating a Machine Learning (ML) model to predict sale prices of real estate properties in CA. The project is divided into four stages: data collection, data analysis, machine learning model creation and API deployment. 



## Data Collection and Cleaning

This stage involved collecting the real estate data using a web scraper. The collected data was then cleaned to improve the accuracy of the analysis. The data cleaning process included the following steps:

1. Removing parameters that were irrelevant, contained constant values, or had a small sample size.
2. Removing rows that contained missing values on important parameters such as price or netHabitableSurface.
3. Removing duplicate rows based on matching latitude, longitude, price, street, postalCode, and number.
4. Imputing missing values for certain parameters.

## Data Analysis

The second stage focused on exploring the dataset through data visualization. Price correlations and price distributions across different regions of Belgium were analyzed after removing outliers. The visualizations and their interpretations can be found in the notebook ` from-data-to-model.ipynb`.

## Machine Learning Model Creation

During the third stage of the project, two types of machine learning models were implemented to predict property prices: Linear Regression and XGBoost and each were tested their performances with different sets of features. After testing, we created a pipeline.
The steps involved in this stage are as follows:

1. Data Splitting: The dataset is split into training and testing datasets for model evaluation. The data is also split based on the type of property (House or Apartment).

2. Feature Scaling: The numerical features in the dataset are scaled using Min-Max Scaler to ensure that all features have the same scale, which improves the performance of the models.

3. Feature encoding: The categorical features in the dataset are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix. This encoding is needed for feeding categorical data to the model as the model used only accepts numerical values.

4. Model Training: A Linear Regression model and an XGBoost model was trained on the training data.

5. Model Evaluation: The performance of our models were evaluated on the testing data using Mean Absolute Error (MAE). The Random Forest Regression model was selected based on it's superior performances compared to the Linear Regression model.

## Installation and Usage

If you dont have Docker installed, you can download and install it from [here](https://www.docker.com/)

To run the FastAPI server, follow these steps:

1. Clone the [xyz](https://github.com/jamesczq/housing-prices-app) repository
2. Navigate to the root of the repository
3. Build the Docker image: `docker build -t house-price-api-image .`
4. Run the Docker container: `docker run --name api_container -p 8000:8000 house-price-api-image`
5. The FastAPI server will be accessible at `http://localhost:8000`

## API Reference

You can interact with the API using the Swagger UI documentation. The Swagger UI is automatically generated and can be accessed from the /docs endpoint.

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

### Endpoint: /predict

```
  POST /predict
```

The POST method expects a JSON object representing the property details for which the price prediction is required. The JSON body should follow the schema described below:

```json
{
{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}
}
```

#### Response:

If the request is successful, the API will return a JSON object containing the predicted property price:

```json
{
    "prediction": 438413,
    "status_code": 200
}
```
