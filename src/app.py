# import the libraries
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

### **Predicting the cost of health insurance for a person**

# Read the data from file using read_csv
medical_insurance_df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
medical_insurance_df
### **Step 2:** Exploratory Data Analysis
# Dataframe information
medical_insurance_df.info()

# duplicate rows
medical_insurance_df.duplicated().sum()
# drop duplicates
medical_insurance_df.drop_duplicates(inplace=True)
# describe dataframe
medical_insurance_df.describe(include='all')

# Visualize the categorical variables with countplot
# Create the figure and axes objects
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Set the palette colors
palette = "blend:#0097CD,#FFB718"

# Create the countplot for sex
sns.countplot(ax=ax[0], data=medical_insurance_df, x='sex', palette=palette)
# Create the countplot for smoker
sns.countplot(ax=ax[1], data=medical_insurance_df, x='smoker', palette=palette)
# Create the countplot for region
sns.countplot(ax=ax[2], data=medical_insurance_df, x='region', palette=palette)

# Show the plot
plt.show()

# Visualize the numerical variables with histogram and boxplot
# Create the figure and axes objects
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# Set the colors
color = sns.color_palette(palette, 4)
# Create the histogram for age
sns.histplot(ax=ax[0, 0], data=medical_insurance_df, x='age', kde=True, color=color[0])
# Create the boxplot for age
sns.boxplot(ax=ax[1, 0], data=medical_insurance_df, x='age', color=color[0])
# Create the histogram for bmi
sns.histplot(ax=ax[0, 1], data=medical_insurance_df, x='bmi', kde=True, color=color[1])
# Create the boxplot for bmi
sns.boxplot(ax=ax[1, 1], data=medical_insurance_df, x='bmi',color=color[1])
# Create the histogram for children
sns.histplot(ax=ax[0, 2], data=medical_insurance_df, x='children', kde=True, color=color[2])
# Create the boxplot for children
sns.boxplot(ax=ax[1, 2], data=medical_insurance_df, x='children', color=color[2])
# Create the histogram for charges
sns.histplot(ax=ax[0, 3], data=medical_insurance_df, x='charges', kde=True, color=color[3])
# Create the boxplot for charges
sns.boxplot(ax=ax[1, 3], data=medical_insurance_df, x='charges', color=color[3])

# Show the plot
plt.show()

# Multivariate analysis
# Encode the categorical variables to numerical values
medical_insurance_df['sex_n'] = pd.factorize(medical_insurance_df['sex'])[0]
medical_insurance_df['smoker_n'] = pd.factorize(medical_insurance_df['smoker'])[0]
medical_insurance_df['region_n'] = pd.factorize(medical_insurance_df['region'])[0]

# Create a reglot and heatmap to visualize the correlation between variables
# Create the figure and axes objects
fig, ax = plt.subplots(4, 3, figsize=(20, 20))

# Define your custom colors as real numbers (RGB values)
color1 = mcolors.to_rgba("#0097CD")
color2 = mcolors.to_rgba("#FFB718")

# Create a custom colormap using the specified colors
custom_cmap = mcolors.ListedColormap([color1, color2])

# Create the regplot and heatmap for sex_n
sns.regplot(ax = ax[0, 0], data = medical_insurance_df, x = 'sex_n', y = 'charges', color = color[0])
sns.heatmap(ax = ax[1, 0], data = medical_insurance_df[['charges', 'sex_n']].corr(), annot = True, cmap = custom_cmap)
# Create the regplot and heatmap for smoker_n
sns.regplot(ax = ax[0, 1], data = medical_insurance_df, x = 'smoker_n', y = 'charges', color = color[0])
sns.heatmap(ax = ax[1, 1], data = medical_insurance_df[['charges', 'smoker_n']].corr(), annot = True, cmap = custom_cmap)
# Create the regplot and heatmap for region_n
sns.regplot(ax = ax[0, 2], data = medical_insurance_df, x = 'region_n', y = 'charges', color = color[0])
sns.heatmap(ax = ax[1, 2], data = medical_insurance_df[['charges', 'region_n']].corr(), annot = True, cmap = custom_cmap)
# Create the regplot and heatmap for age
sns.regplot(ax = ax[2, 0], data = medical_insurance_df, x = 'age', y = 'charges', color = color[0])
sns.heatmap(ax = ax[3, 0], data = medical_insurance_df[['charges', 'age']].corr(), annot = True, cmap = custom_cmap)
# Create the regplot and heatmap for bmi
sns.regplot(ax = ax[2, 1], data = medical_insurance_df, x = 'bmi', y = 'charges', color = color[0])
sns.heatmap(ax = ax[3, 1], data = medical_insurance_df[['charges', 'bmi']].corr(), annot = True, cmap = custom_cmap)
# Create the regplot and heatmap for children
sns.regplot(ax = ax[2, 2], data = medical_insurance_df, x = 'children', y = 'charges', color = color[0])
sns.heatmap(ax = ax[3, 2], data = medical_insurance_df[['charges', 'children']].corr(), annot = True, cmap = custom_cmap)

# Show the plot
plt.show()

# Feature engineering
# Scale the variables using MinMaxScaler

# Fit the data
numeric_columns = ['sex_n', 'smoker_n', 'region_n', 'age', 'bmi', 'children', 'charges']

# Create the scaler object
scaler = MinMaxScaler()

# Scale the variables
scale_features = scaler.fit_transform(medical_insurance_df[numeric_columns])
medical_insurance_scaled_df = pd.DataFrame(scale_features, columns=numeric_columns)
medical_insurance_scaled_df.head()

# Feature selection
# Separate the feature and the target columns
X = medical_insurance_scaled_df.drop(columns=['charges'])
y = medical_insurance_scaled_df['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the best features
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train, y_train)

# Create a dataframe with the features
X_train_sel = pd.DataFrame(selector.transform(X_train), columns=X_train.columns[selector.get_support()])
X_test_sel = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()])

# Show the results
X_train_sel.head()
X_test_sel.head()
# Add the target column to the selected features
X_train_sel['charges'] = y_train.values
X_test_sel['charges'] = y_test.values

# Save the selected features to a csv file
X_train_sel.to_csv('../data/processed/train.csv', index=False)
X_test_sel.to_csv('../data/processed/test.csv', index=False)
### **Step 3:** Build the linear regression model
# Read the data from file
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

# Show the first rows of the train dataframe
train_df.head()
# Separate the feature and the target columns
X_train = train_df.drop(columns=['charges'])
y_train = train_df['charges']

# Separate the feature and the target columns
X_test = test_df.drop(columns=['charges'])
y_test = test_df['charges']

# Create the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Know the parameters of the model
print('Intercept:', linear_model.intercept_)
print('Coefficients:', linear_model.coef_)
# Predict the values
y_pred = linear_model.predict(X_test)
y_pred

# Evaluate the model
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# Calculate the root mean squared error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print('Root mean squared error:', rmse)

# Calculate the r squared score
r2 = r2_score(y_test, y_pred)
print('R squared score:', r2)