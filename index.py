# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Covid 19.csv"

file_path ='Covid 19.csv' 
df = pd.read_csv(file_path)

# Display first 10 rows
print("First 10 rows of the dataset:")
print(df.head(10))

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Convert 'As of Date' to datetime
df['Week End'] = pd.to_datetime(df['Week End'], errors='coerce')

# -------------------------------
# New Objective: Calculate Risk Ratio
# -------------------------------

# Step 1: Select important columns
df_risk = df[['Age Group', 'Outcome', 'Unvaccinated Rate', 'Vaccinated Rate']].dropna()

# Step 2: Calculate Risk Ratio
df_risk['Risk Ratio (Unvaccinated/Vaccinated)'] = df_risk['Unvaccinated Rate'] / df_risk['Vaccinated Rate']

# Step 3: Average Risk Ratio by Age Group and Outcome
risk_summary = df_risk.groupby(['Age Group', 'Outcome'])['Risk Ratio (Unvaccinated/Vaccinated)'].mean().reset_index()

print("\nAverage Risk Ratio (Unvaccinated vs Vaccinated) by Age Group and Outcome:")
print(risk_summary)

# Step 4: Plot Risk Ratio
plt.figure(figsize=(14, 7))
sns.barplot(data=risk_summary, x='Age Group', y='Risk Ratio (Unvaccinated/Vaccinated)', hue='Outcome', palette='coolwarm')
plt.title('Risk of Unvaccinated vs Vaccinated by Age Group and Outcome')
plt.ylabel('Risk Ratio (Higher means More Risk for Unvaccinated)')
plt.xlabel('Age Group')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()

# -------------------------------
# Other Previous Graphs
# -------------------------------

# Filter deaths
deaths_df = df[df['Outcome'] == 'Deaths']

# 1. Average Unvaccinated Death Rate
average_unvaccinated_death_rate = np.nanmean(deaths_df['Unvaccinated Rate'])
print(f"\nAverage Unvaccinated Death Rate: {average_unvaccinated_death_rate:.2f}")

# 2. Bar Plot: Death Rate Comparison
deaths_df_grouped = deaths_df.groupby('Age Group')[['Unvaccinated Rate', 'Vaccinated Rate']].mean().reset_index()
deaths_df_melted = deaths_df_grouped.melt(id_vars='Age Group', value_vars=['Unvaccinated Rate', 'Vaccinated Rate'],
                                          var_name='Vaccination Status', value_name='Rate')

plt.figure(figsize=(12, 6))
sns.barplot(data=deaths_df_melted, x='Age Group', y='Rate', hue='Vaccination Status')
plt.title('Unvaccinated vs Vaccinated Death Rate by Age Group')
plt.xticks(rotation=45)
plt.ylabel('Death Rate')
plt.xlabel('Age Group')
plt.tight_layout()
plt.show()

# 3. Pie Chart: Total Outcomes
unvaccinated_total = deaths_df['Outcome Unvaccinated'].sum()
vaccinated_total = deaths_df['Outcome Vaccinated'].sum()

plt.figure(figsize=(6, 6))
plt.pie(
    [unvaccinated_total, vaccinated_total],
    labels=['Unvaccinated', 'Vaccinated'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#ff9999', '#66b3ff']
)
plt.title('Overall Outcome Distribution (Deaths)')
plt.axis('equal')
plt.show()

# 4. Histogram: Unvaccinated Death Rate Distribution
plt.figure(figsize=(8, 5))
plt.hist(deaths_df['Unvaccinated Rate'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Unvaccinated Death Rates')
plt.xlabel('Unvaccinated Death Rate')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. Line Plot: Trend of Death Rates
deaths_time_df = df[df['Outcome'] == 'Deaths'].sort_values('Week End')

plt.figure(figsize=(14, 6))
sns.lineplot(data=deaths_time_df, x='Week End', y='Unvaccinated Rate', label='Unvaccinated Rate', color='red')
sns.lineplot(data=deaths_time_df, x='Week End', y='Vaccinated Rate', label='Vaccinated Rate', color='green')
plt.title('Trend of Death Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Death Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Box Plot: Death Rate by Age Group
plt.figure(figsize=(12, 6))
sns.boxplot(data=deaths_df, x='Age Group', y='Unvaccinated Rate')
plt.title('Distribution of Unvaccinated Death Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Unvaccinated Death Rate')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
