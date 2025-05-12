import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
iris = load_iris(as_frame=True)
df = iris.frame

print("First few rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nChecking for missing values:")
print(df.isnull().sum())

# The Iris dataset is quite clean, so no missing values to handle in this case.

# Task 2: Basic Data Analysis
print("\nBasic statistics of numerical columns:")
print(df.describe())

print("\nMean of numerical columns grouped by species:")
print(df.groupby('target')['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'].mean())

print("\nInteresting findings:")
print("- The average petal length and width seem to vary significantly between the different species.")
print("- Setosa generally has smaller petal dimensions but a relatively wider sepal compared to the other two species.")
print("- Versicolor and virginica have overlapping ranges for some features, suggesting they might be harder to distinguish based on these measurements alone.")

# Task 3: Data Visualization
#No.1
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
plt.title('Features of Iris Flowers')
plt.xlabel('Data Point Index')
plt.ylabel('Centimeters (cm)')
plt.legend(title='Feature')
plt.grid(True)
plt.show()

#No.2
plt.figure(figsize=(8, 6))
sns.barplot(x='target', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks([0, 1, 2], iris.target_names)
plt.show()

#No.3
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

#No.4
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', labels=iris.target_names)
plt.grid(True)
plt.show()
