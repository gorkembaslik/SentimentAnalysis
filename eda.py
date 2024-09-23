import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv('youtube_data.csv')

# View the first few rows of the dataset
print(df.head())

# Get basic information about the dataset
df.info()

# Check for missing values
print(df.isnull().sum())

# Get the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Get summary statistics for numeric columns
print(df.describe())

# Plot distributions of views, likes, dislikes, comments, and sentiment score
plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)
sns.histplot(df['Views'], bins=30, kde=True)
plt.title('Distribution of Views')

plt.subplot(2, 3, 2)
sns.histplot(df['Likes'], bins=30, kde=True)
plt.title('Distribution of Likes')

#plt.subplot(2, 3, 3)
#sns.histplot(df['Dislikes'], bins=30, kde=True)
#plt.title('Distribution of Dislikes')

plt.subplot(2, 3, 3)
sns.histplot(df['Comments'], bins=30, kde=True)
plt.title('Distribution of Comments')

plt.subplot(2, 3, 4)
sns.histplot(df['Sentiment_Score'], bins=30, kde=True)
plt.title('Distribution of Sentiment Score')

plt.tight_layout()
plt.show()

# Select only the numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Create a correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of Views vs Likes
plt.figure(figsize=(8,6))
sns.scatterplot(x='Views', y='Likes', data=df)
plt.title('Views vs Likes')
plt.show()

# Scatter plot of Views vs Sentiment_Score
plt.figure(figsize=(8,6))
sns.scatterplot(x='Views', y='Sentiment_Score', data=df)
plt.title('Views vs Sentiment Score')
plt.show()

# Group by 'Sponsored' column and calculate the mean for relevant metrics
numeric_cols = ['Views', 'Likes', 'Comments', 'Sentiment_Score']
sponsored_group = df.groupby('Sponsored')[numeric_cols].mean()
print(sponsored_group)

# Plot the comparison of sponsored vs unsponsored videos
sponsored_group.plot(kind='bar', figsize=(10,6))
plt.title('Sponsored vs Unsponsored Videos: Average Metrics')
plt.ylabel('Average Values')
plt.show()

# Plot sentiment score distribution for sponsored and unsponsored videos
plt.figure(figsize=(10,6))
sns.boxplot(x='Sponsored', y='Sentiment_Score', data=df)
plt.title('Sentiment Score Distribution for Sponsored vs Unsponsored Videos')
plt.show()

# Create new engagement metrics
df['Comments_Views_Ratio'] = df['Comments'] / df['Views']

# View basic statistics for these new ratios
print(df['Comments_Views_Ratio'].describe())

# Plot the distributions of these ratios
plt.figure(figsize=(10,6))

plt.subplot(1, 2, 1)
sns.histplot(df['Comments_Views_Ratio'], bins=30, kde=True)
plt.title('Comments-Views Ratio Distribution')

plt.tight_layout()
plt.show()

