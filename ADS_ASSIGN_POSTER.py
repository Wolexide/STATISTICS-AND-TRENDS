

#import the standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score

# define a function to extract the data and the transposed dataframe
def extracted_data(file_path, sheet_name, columns):
    """
    The necessary parameters are
    file_path : this is the url of the world bank data
    countries : list of countries to be analysed.
    columns : list of years to be used.
    """
    df = pd.read_excel(file_path, sheet_name= sheet_name, skiprows=3)
    df = df.drop(columns=columns)
    df.set_index('Country Name', inplace=True)
    return df, df.T


# the parameters are passed into the function
file_path= 'https://api.worldbank.org/v2/en/indicator/EG.ELC.ACCS.ZS?downloadformat=excel'
sheet_name = 'Data'
columns = ['Country Code', 'Indicator Name', 'Indicator Code']
electricity, electricity_transpose = extracted_data(file_path, sheet_name, columns)
electricity


data_electricity = electricity.loc[:, ['1999','2020']]
print(data_electricity)


# check for null values
data_electricity.isnull().sum()

# drop null values
data_clustering = data_electricity.dropna()
print(data_clustering)


# change data to array
x_values = data_clustering.values
x_values

# plot scatterplot of data_clustering 
plt.figure(figsize=(12,10))
plt.scatter(data_clustering['1999'], data_clustering['2020'])
plt.title('Scatterplot of electricity between 1999 and 2020', fontsize=20)
plt.xlabel('Year 1999', fontsize=15)
plt.ylabel('Year 2020', fontsize=15)
plt.show()

# x_values is scaled for clustering
min_val = np.min(x_values)
max_val = np.max(x_values)
x_norm = (x_values-min_val) / (max_val-min_val)
print(x_norm)

# the best number of clusters is chosen using sum of squared error
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=10, random_state=3)
    kmeans.fit(x_norm) # the normalised data is fit using KMeans method
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xlabel('clusters')
plt.ylabel('sse')
plt.show()

# according to the elbow, 2 is the best number for clustering
kmeans = KMeans(n_clusters=2, random_state=3)
y_pred = kmeans.fit_predict(x_norm)


# the centroid of the clusters are determined
cluster_center = kmeans.cluster_centers_
print(cluster_center)


# the silhouette score is determined
silhouette = silhouette_score(x_norm, y_pred)
print(silhouette)

# this plot shows the clusters and centroids using matplotlib
plt.figure(figsize=(12,8))
plt.scatter(x_norm[y_pred == 0, 0], x_norm[y_pred == 0, 1], s = 50, c = 'blue', label = 'cluster 0')
plt.scatter(x_norm[y_pred == 1, 0], x_norm[y_pred == 1, 1], s = 50, c = 'red', label = 'cluster 1')
plt.scatter(cluster_center[:, 0], cluster_center[:,1], s = 150, c = 'black', label = 'Centroids')
plt.title('Scatterplot showing the clusters and centroids of electricity in countries of the world from 1999 to 2020', fontsize=20)
plt.xlabel('1999', fontsize=15)
plt.ylabel('2020', fontsize=15)
plt.legend()
plt.show()

# the clusters are stored in a column called cluster
data_clustering['cluster'] = y_pred
df_label = data_clustering.loc[data_clustering['cluster'] == 1] # the first cluster label 
df_bar = df_label.iloc[:5,] # the first 6 countries of the third cluster are used for analysis
print(df_bar)

# plot a multiple bar plot showing different countries across two years
array_label = ['Africa Western and Central', 'Burundi', 'Benin', 'Burkina Faso', 'Bangladesh']
x = np.arange(len(array_label)) # x is the range of values using the length of the label_array
width = 0.2
fig, ax  = plt.subplots(figsize=(12,10))
    
plt.bar(x - width, df_bar['1999'], width, label='1999') 
plt.bar(x, df_bar['2020'], width, label='2020')   
plt.title('% Electricity production across five countries between 1999 and 2020', fontsize=15)
plt.ylabel('% electricity', fontsize=15)
plt.xticks(x, array_label)
plt.legend()
ax.tick_params(bottom=False, left=True)

plt.show()

# the transpose data is sliced for easy analysis
df_fit = electricity_transpose.iloc[-22:-1]
df_fit

# a dataframe is created using angola as a point of discussion
df_Angola = pd.DataFrame({
    'Year' : df_fit.index,
    'Angola' : df_fit['Angola']
})
df_Angola.reset_index(drop=True)

# the year column is converted to integer
df_Angola['Year'] = np.array(df_Angola['Year']).astype(int)

# this is a fitting function
def model_fit(x, a, b, c, d):
    '''
    The function calculates the polynomial function which accepts some parameters:
    x: these are the years of the data column
    a,b,c,d are constants
    
    '''
    y = a*x**3 + b*x**2 + c*x + d
    return y

# the parameters are passed into the curve_fit method and the parameters and covariance are calculated 
param, cov = curve_fit(model_fit, df_Angola['Year'], df_Angola['Angola'])
print(param)
print(cov)

# a range of values for years is created for prediction
year = np.arange(2000, 2036)
predict = model_fit(year, *param) # the params and years are passed into the model_fitting function and the predictions are calculated 

# this is a plot of electricity in Angola and the predictions for the next 16 years 
plt.figure(figsize=(12,10))
plt.plot(df_Angola["Year"], df_Angola["Angola"], label="Angola")
plt.plot(year, predict, label="predictions")
plt.title('A Plot showing the predictions of electricity production in Angola', fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("% electricity", fontsize=15)
plt.legend()
plt.show()

# Here the predictions for the next 16 years are put in a dataframe 
df_predict = pd.DataFrame({'Year': year, 'predictions': predict})
df_sixteen_years_prediction = df_predict.iloc[21:,]
print(df_sixteen_years_prediction)



