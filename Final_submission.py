# ## Overview
# The dataset contains 947 rows and 8 columns. After data validation, some modifications were made to handle missing values and ensure consistency in the data. This report provides an overview of the data validation process and the changes made.
# 
# ### Data Validation
# - **Recipe:** There are 947 unique identifiers without any missing values. After dataset cleaning, 52 rows were removed due to missing values in other columns.
# 
# - **Calories:** There are 895 non-null values. Dropped 52 NaN.
# 
# - **Carbohydrate:** There are 895 non-null values. Dropped 52 NaN.
# 
# - **Sugar:** There are 895 non-null values. Dropped 52 NaN.
# 
# - **Protein:** There are 895 non-null values. Dropped 52 NaN.
# 
# - **Category:** There are 11 unique values without any missing values. It was found that there was an additional value "Chicken Breast," which was united with the "Chicken" category.
# 
# - **Servings:** There are 6 unique values without any missing values. The column type was changed to integer, and two extra values "4 as a snack" and "6 as a snack" were united with "4" and "6," respectively.
# 
# - **High Traffic:** There are 574 non-values that correspond ("High"). The 360 null values were replaced with "Low."
# 
# ### Data Cleaning
# - Rows with missing values in "calories," "carbohydrate," "sugar," and "protein" columns were removed to maintain data integrity.
# 
# - "Chicken Breast" category was united with the "Chicken" category to ensure consistency in the categories.
# 
# - Extra values "4 as a snack" and "6 as a snack" in the "servings" column were united with "4" and "6," respectively, and the column type was changed to integer.

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
plt.style.use('ggplot')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Import the dataset
df = pd.read_csv('recipe_site_traffic_2212.csv')
df.info()
df.head()

# Data Validation
print(df.isna().sum().sort_values())
df.dropna(subset=['calories', 'carbohydrate', 'sugar', 'protein'], inplace=True)
df['high_traffic'].fillna(value='Low', inplace=True)

# Data Cleaning
df['servings'] = df['servings'].str.replace(' as a snack','')
df['servings'] = df['servings'].astype(int
df['category'] = df['category'].str.replace('Chicken Breast', 'Chicken').astype('category')
df['high_traffic'] = df['high_traffic'].astype('category')

# Exploratory Analysis
# Numeric Features
fig = plt.figure(figsize=(20, 22), constrained_layout=True)
spec = fig.add_gridspec(4, 2)

# calories
ax00 = fig.add_subplot(spec[0, 0])
sns.histplot(df['calories'], color='gray',ax=ax00).set(title='Distribution of recipes by the calories')

ax01 = fig.add_subplot(spec[0, 1])
sns.boxplot(x='calories', data=df, color='#7E1037', ax=ax01).set(title='Calories Distribution')

# protein
ax10 = fig.add_subplot(spec[1, 0])
sns.histplot(df['protein'], color='gray',ax=ax10).set(title='Distribution of recipes by the protein')

ax11 = fig.add_subplot(spec[1, 1])
sns.boxplot(x='protein', data=df, color='#7E1037', ax=ax11).set(title='Protein Distribution')

# sugar
ax20 = fig.add_subplot(spec[2, 0])
sns.histplot(df['sugar'], color='gray',ax=ax20).set(title='Distribution of recipes by the sugar')

ax21 = fig.add_subplot(spec[2, 1])
sns.boxplot(x='sugar', data=df, color='#7E1037', ax=ax21).set(title='Sugar Distribution')

# carbohydrate
ax30 = fig.add_subplot(spec[3, 0])
sns.histplot(df['carbohydrate'], color='gray',ax=ax30).set(title='Distribution of recipes by the carbohydrate')

ax31 = fig.add_subplot(spec[3, 1])
sns.boxplot(x='carbohydrate', data=df, color='#7E1037', ax=ax31).set(title='Carbohydrate Distribution')

# set the suptitle
_=fig.suptitle('Distribution of Numeric features',
               fontsize=20,
               fontweight='bold')

plt.show()



# Handle Outliers
data_before = df.copy()

# Handling Outliers Function
def handle_outlier(df, col):
    Q3 = df[col].quantile(0.75)
    Q1 = df[col].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + (1.5 * IQR)
    lower = Q1 - (1.5 * IQR)
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# Specify the columns to handle and visualize
data_num = df.select_dtypes(include = ["float64", "int64"])

# Apply the outlier handling function only to the specified columns
for column in data_num.columns:
    handle_outlier(df, column)

sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data_before,ax=axes[0])
sns.boxplot(df,ax=axes[1])
axes[0].set_title("Before Handling Outliers")
axes[1].set_title("After Handling Outliers")
fig.suptitle("The Distribution of Numeric Variables")
axes[0].set_xlabel("Numeric Variables")
axes[1].set_xlabel("Numeric Variables")
axes[0].set_ylabel("Amount")
axes[1].set_ylabel("Amount")
plt.tight_layout()
plt.show()

# Heatmap
fig, ax = plt.subplots(figsize=(8,6))  
sns.heatmap(df[['calories', 'protein', 'carbohydrate', 'sugar', 'servings']].corr(), annot=True)
plt.show()

# Numeric Feature vs Target Variable
# plot the boxplots of the nutritional characteristics of the recipes for high
# traffic and low traffic

fig = plt.figure(figsize=(10, 12), constrained_layout=True)
spec = fig.add_gridspec(2, 2)

# calories
ax00 = fig.add_subplot(spec[0, 0])
sns.boxplot(x='high_traffic', y='calories', data=df, ax=ax00).set(title='Calories content by high traffic')

ax01 = fig.add_subplot(spec[0, 1])
sns.boxplot(x='high_traffic', y='protein', data=df, ax=ax01).set(title='Protein content by high traffic')

# sugar
ax10 = fig.add_subplot(spec[1, 0])
sns.boxplot(x='high_traffic', y='sugar', data=df, ax=ax10).set(title='Sugar content by high traffic')

ax11 = fig.add_subplot(spec[1, 1])
sns.boxplot(x='high_traffic', y='carbohydrate', data=df, ax=ax11).set(title='Carbohydrate content by high traffic')

# set the suptitle
_=fig.suptitle('Numeric features by high traffic',
               fontsize=20,
               fontweight='bold')

plt.show()

# Categorical Feature
sns.histplot(df['category'], color='gray').set(title='Distribution of recipes by the category')
plt.xticks(rotation=90)
plt.show()



# Model Fitting and Evaluation
# Encoding Categorical Data 
df = pd.get_dummies(df, columns=['category'])
labelencoder = LabelEncoder()
df['high_traffic'] = labelencoder.fit_transform(df['high_traffic'])

# Split and Scaling
columns_to_std = ['calories', 'carbohydrate', 'sugar','protein']
df['std'] = df[columns_to_std].apply(lambda row: np.std(row), axis=1)

column_order = ['calories',  'carbohydrate',   'sugar',  'protein','std' , 'servings','category_Beverages'  ,'category_Breakfast' , 'category_Chicken',  'category_Dessert' , 'category_Lunch/Snacks' ,'category_Meat',  'category_One Dish Meal',  'category_Pork',  'category_Potato', 'category_Vegetable' ,'high_traffic'] 
df = df[column_order]


X = df.drop(['high_traffic'], axis=1).values
y = df['high_traffic'].values


numeric_features = X[:, :5]

# Standardization
st_scaler = StandardScaler()
st_scaler.fit(numeric_features)
numeric_features_scaler = st_scaler.transform(numeric_features)
X_scaler = np.concatenate((numeric_features_scaler, X[:, 5:]), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size= 0.20, random_state=10)

# Logistic Regression
param_grid = {
    'C': np.logspace(-4, 4, 20),  # Regularization strength (inverse)
    'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],  # Optimization algorithm
    'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization type
    'max_iter': [100, 200, 300, 600],  # Max iterations for solver
    'tol': [1e-4, 5e-4, 6e-4],  # Tolerance for stopping criteria
    'class_weight': ['balanced', None], 
    'class_weight': [None, 'balanced']  
}
cv = StratifiedKFold(n_splits=6,  shuffle=True)
logistic_model = LogisticRegression(solver='liblinear', class_weight='balanced')

model = RandomizedSearchCV(
    logistic_model, 
    param_grid, 
    n_iter=100, 
    scoring='recall',        # Use F1 score as the evaluation metric
    cv=cv,        
    n_jobs=-1
)

model.fit(X_train, y_train)
best_model = model.best_estimator_
print("Best Parameters:", model.best_params_)

y_pred = best_model.predict(X_test)

# evaluate the accuracy, precision and recall of the logistic regression model
accuracy_lr = accuracy_score(y_test, y_pred)
print(accuracy_lr)
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)


# Feature Importance
importances = pd.Series(best_model.coef_[0],index=column_order[:-1])
sorted_importances=importances.sort_values()
plt.figure(figsize=(10, 6))
sorted_importances.plot(kind='barh')
plt.title('Feature Importance in Logistic Regression Model')
plt.show()


# Random Forest

param_grid_rf = {
    'n_estimators': [50, 100, 200, 300, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
    'bootstrap': [True, False],  # Whether to use bootstrap samples for building trees
    'criterion': ['gini', 'entropy'],  # Split quality measure
    'max_samples': [None, 0.8, 0.9],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(class_weight='balanced')
cv = StratifiedKFold(n_splits=6, shuffle=True)
random_search = RandomizedSearchCV(estimator=rf, 
                                      param_distributions=param_grid_rf, 
                                      n_iter=100, cv=cv, verbose=True, n_jobs=-1)
# Fit the model
random_search.fit(X_train, y_train)
best_model_rf = random_search.best_estimator_

# Get the best parameters
best_params = random_search.best_params_
print("Best Parameters: ", best_params)

y_pred_rf = best_model_rf.predict(X_test)

# evaluate the accuracy, precision and recall of the logistic regression model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)


# Results
results = {
    'Classifier': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_rf],
    'Precision': [precision_lr, precision_rf],
    'Recall': [recall_lr, recall_rf]
}

results_df = pd.DataFrame(results)

results_df

