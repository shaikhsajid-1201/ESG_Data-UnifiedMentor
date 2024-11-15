import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('Dataset/ESGData.csv')

# Display the first few rows to verify the data
print(df.head())

# Show all columns
print(df.columns.tolist())

# Check the shape of the dataset
print("Dataset shape:", df.shape)

# Check if there are any other columns that might identify what each row represents
print(df.iloc[:5, :6])  # Show first 5 rows and first 6 columns

# Calculate percentage of null values in each column
null_percentage = df.isnull().sum() / len(df) * 100

# Keep only year columns (those that can be converted to integers) and have less than 40% null values
year_columns = [col for col in df.columns if col.isdigit() and null_percentage[col] < 40]

# Keep non-year columns (metadata columns) plus the valid year columns
columns_to_keep = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + year_columns
df = df[columns_to_keep]

# Verify the shape of the cleaned dataset
print("Final shape:", df.shape)
print("\nYears kept:", sorted(year_columns))
print("\nRemaining null values percentage:\n", (df.isnull().sum() / len(df) * 100).round(2))

# First melt the DataFrame to convert years from columns to rows
df_melted = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name='Year',
                    value_name='Value')

# Convert Year to integer type
df_melted['Year'] = df_melted['Year'].astype(int)

# Sort by Year and other relevant columns
df_melted = df_melted.sort_values(['Year', 'Country Name', 'Indicator Name'])

# Display the first few rows to verify the transformation
print("\nTransposed DataFrame:")
print(df_melted.head())
print("\nNew shape:", df_melted.shape)

# Calculate mean value for each indicator
indicator_means = df_melted.groupby('Indicator Name')['Value'].mean().sort_values(ascending=False)

# Get top 3 indicators
top_3_indicators = indicator_means.head(3).index.tolist()

# Filter data for top 3 indicators
top_3_df = df_melted[df_melted['Indicator Name'].isin(top_3_indicators)]

# Create the plot
plt.figure(figsize=(15, 8))
sns.lineplot(data=top_3_df, x='Year', y='Value', hue='Indicator Name')

# Customize the plot
plt.title('Top 3 ESG Indicators Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print the top 3 indicators and their mean values
print("\nTop 3 ESG Indicators by Average Value:")
for indicator, mean_value in indicator_means.head(3).items():
    print(f"{indicator}: {mean_value:.2f}")

# 1. Trend Analysis for Bottom 3 Indicators
bottom_3_indicators = indicator_means.tail(3).index.tolist()
bottom_3_df = df_melted[df_melted['Indicator Name'].isin(bottom_3_indicators)]

# Check if data exists
if not bottom_3_df.empty:
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=bottom_3_df, x='Year', y='Value', hue='Indicator Name')
    plt.title('Bottom 3 ESG Indicators Over Time', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print the indicators being plotted
    print("\nBottom 3 ESG Indicators being plotted:")
    for indicator, mean_value in indicator_means.tail(3).items():
        print(f"{indicator}: {mean_value:.2f}")
else:
    print("No data available for bottom 3 indicators")

# 2. Regional Analysis for Top Indicators
# Group by regions (you might want to create a region mapping)
top_countries = df_melted[df_melted['Indicator Name'] == top_3_indicators[0]].groupby('Country Name')['Value'].mean().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title(f'Top 10 Countries for {top_3_indicators[0]}', fontsize=14)
plt.xlabel('Average Value', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Year-over-Year Change Analysis
# Calculate year-over-year change for top indicator
top_indicator_df = df_melted[df_melted['Indicator Name'] == top_3_indicators[0]].copy()
top_indicator_df['YoY_Change'] = top_indicator_df.groupby('Country Name')['Value'].pct_change() * 100

plt.figure(figsize=(15, 6))
sns.boxplot(data=top_indicator_df, x='Year', y='YoY_Change')
plt.title(f'Year-over-Year Change Distribution for {top_3_indicators[0]}', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Percentage Change (%)')
plt.tight_layout()
plt.show()

# 4. Correlation Analysis between Top Indicators
pivot_df = df_melted.pivot_table(
    index=['Country Name', 'Year'],
    columns='Indicator Name',
    values='Value'
).reset_index()

correlation = pivot_df[top_3_indicators].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Top 3 Indicators', fontsize=14)
plt.tight_layout()
plt.show()

# 5. Summary Statistics
print("\nSummary Statistics for Top 3 Indicators:")
for indicator in top_3_indicators:
    indicator_stats = df_melted[df_melted['Indicator Name'] == indicator]['Value'].describe()
    print(f"\n{indicator}:")
    print(indicator_stats)

# 6. Volatility Analysis
volatility = df_melted.groupby('Indicator Name')['Value'].std().sort_values(ascending=False)
print("\nIndicator Volatility (Standard Deviation):")
print(volatility.head())

# 7. Missing Data Analysis by Year
yearly_completeness = df_melted.groupby('Year').apply(
    lambda x: (x['Value'].notna().sum() / len(x)) * 100
).round(2)

plt.figure(figsize=(15, 6))
yearly_completeness.plot(kind='line', marker='o')
plt.title('Data Completeness by Year (%)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Completeness (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8. Recent Trends (Last 5 Years)
recent_years = df_melted['Year'].astype(int).max() - 5
recent_trends = df_melted[
    (df_melted['Year'].astype(int) >= recent_years) & 
    (df_melted['Indicator Name'].isin(top_3_indicators))
]

# Line plot for trends
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=recent_trends,
    x='Year',
    y='Value', 
    hue='Indicator Name',
    style='Indicator Name',
    markers=True,
    dashes=False
)
plt.title('Recent Trends in Top 3 ESG Indicators', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Box plot to show distribution by year
plt.figure(figsize=(15, 8))
sns.boxplot(
    data=recent_trends,
    x='Year',
    y='Value',
    hue='Indicator Name'
)
plt.title('Distribution of Top 3 ESG Indicators by Year', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
