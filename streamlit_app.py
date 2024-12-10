import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np



# Welcome message
st.title("☀️ Solar Irradiance Analysis Dashboard 🌅")
st.info(""" 
    **Business Objective:**
    MoonLight Energy Solutions aims to develop a strategic approach to significantly enhance its operational efficiency and sustainability through targeted solar investments. As an Analytics Engineer at MoonLight Energy Solutions, your task is to perform a quick analysis of environmental measurements provided by the engineering team and translate your observations into a strategy report. 
    
    Your analysis should focus on identifying key trends and deriving valuable insights that will support your data-driven case. Your recommendations, based on statistical analysis and EDA, should aim at identifying high-potential regions for solar installation that align with the company's long-term sustainability goals. Your report should provide insights to help realize the overarching objectives of MoonLight Energy Solutions.
""")

# List of Authors
st.write("""
    **Author:** Getnet B. Begashaw (PhD Candidate in Statistics)  
    **Email:** [getnetbogale145@gmail.com](mailto:getnetbogale145@gmail.com)
""")



# Function to download and load data
def download_and_load_data(file_name, file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, file_name, quiet=True)
    return pd.read_csv(file_name)

# File IDs from Google Drive
file_ids = {
    "Benin Dataset": ("benin-malanville.csv", "13qbGUG01DQcfTvEZ0N9cotd4WqOAC1lP"),
    "Sierra Leone Dataset": ("sierraleone-bumbuna.csv", "1JQPkpP-gpu_RTay2NiFy7sKLBrfV_zCg"),
    "Togo Dataset": ("togo-dapaong_qc.csv", "1DWXjH-mVYntCEAg3ye60v4Jl3IBcNyDW"),
}

# Function to integrate all datasets
def integrate_datasets(file_ids):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame
    for country, (file_name, file_id) in file_ids.items():
        df = download_and_load_data(file_name, file_id)
        df["Country"] = country  # Add a column to identify the country
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# Automatically load and integrate datasets
with st.spinner("Downloading and integrating datasets..."):
    combined_df = integrate_datasets(file_ids)
st.success("Datasets integrated successfully!")

st.header(" Step 1: Data Preprocessing Pipeline")

# Display integrated dataset in an expander
with st.expander("Integrated Dataset (First 100 rows)"):
    st.dataframe(combined_df.head(100))  # Display only the first 100 rows for efficiency

# Frequency count of 'Country' column
country_frequency = combined_df['Country'].value_counts()

# Expanders for different data views
with st.expander('🔍 Dataset Information'):
    buffer = StringIO()
    combined_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Expander for displaying the frequency
with st.expander("Country Frequency Count"):
    st.write(country_frequency)


with st.expander("Data Type Correction"):
    # Convert 'Timestamp' to datetime format
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')

    # Ensure 'BP' and 'Cleaning' are integers (if necessary)
    # combined_df['BP'] = combined_df['BP'].astype('int64')
    # combined_df['Cleaning'] = combined_df['Cleaning'].astype('int64')

    # Convert 'Country' column to category for memory optimization
    combined_df['Country'] = combined_df['Country'].astype('category')
    combined_df['Cleaning'] = combined_df['Cleaning'].astype('object')

    # Since 'Comments' column contains only NaN, it's likely safe to drop it
    combined_df = combined_df.drop(columns=['Comments'])

    # Check for any other type issues (e.g., non-numeric columns that should be numeric)
    combined_df['GHI'] = pd.to_numeric(combined_df['GHI'], errors='coerce')
    combined_df['DNI'] = pd.to_numeric(combined_df['DNI'], errors='coerce')
    combined_df['DHI'] = pd.to_numeric(combined_df['DHI'], errors='coerce')
    combined_df['ModA'] = pd.to_numeric(combined_df['ModA'], errors='coerce')
    combined_df['ModB'] = pd.to_numeric(combined_df['ModB'], errors='coerce')
    combined_df['Tamb'] = pd.to_numeric(combined_df['Tamb'], errors='coerce')
    combined_df['RH'] = pd.to_numeric(combined_df['RH'], errors='coerce')
    combined_df['WS'] = pd.to_numeric(combined_df['WS'], errors='coerce')
    combined_df['WSgust'] = pd.to_numeric(combined_df['WSgust'], errors='coerce')
    combined_df['WSstdev'] = pd.to_numeric(combined_df['WSstdev'], errors='coerce')
    combined_df['WD'] = pd.to_numeric(combined_df['WD'], errors='coerce')
    combined_df['WDstdev'] = pd.to_numeric(combined_df['WDstdev'], errors='coerce')
    # combined_df['Precipitation'] = pd.to_numeric(combined_df['Precipitation'], errors='coerce')
    combined_df['TModA'] = pd.to_numeric(combined_df['TModA'], errors='coerce')
    combined_df['TModB'] = pd.to_numeric(combined_df['TModB'], errors='coerce')

    # Display the final data types
    st.write("### Final Data Types:")
    st.write(combined_df.dtypes)

with st.expander("Handling Missing Data"):
    # Check for missing values before treatment
    st.write("### Missing values before treatment:")
    missing_values = combined_df.isnull().sum()
    st.write(missing_values)


with st.expander("Encoding Categorical Variables (One-Hot Encoding)"):
    # Use pandas get_dummies to perform one-hot encoding
    data_encoded = pd.get_dummies(combined_df, columns=['Country', 'Cleaning'], drop_first=False)

    # Display the resulting DataFrame (first 5 rows)
    st.write("### One-Hot Encoded Data (First 5 rows):")
    st.dataframe(data_encoded.head())






with st.expander("Outlier Detection & Handling"):
    # Dynamically select continuous columns (numerical columns excluding categorical ones)
    continuous_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the number of columns and adjust the figure size accordingly
    n_cols = len(continuous_columns)
    fig_width = max(15, n_cols * 3)  # Dynamic width based on the number of columns

    # Create a figure with a horizontal scrollable layout
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(fig_width, 12), squeeze=False)

    # Loop through each continuous column and create box plots before and after outlier treatment
    for i, col in enumerate(continuous_columns):
        # Before handling outliers
        sns.boxplot(data=combined_df, x=col, ax=axes[0, i])
        axes[0, i].set_title(f'Before: {col}')
        axes[0, i].set_xlabel('')
        
        # Handle outliers using IQR
        Q1 = combined_df[col].quantile(0.25)
        Q3 = combined_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out rows where the value is outside the IQR range
        data_cleaned = combined_df[(combined_df[col] >= lower_bound) & (combined_df[col] <= upper_bound)]

        # After handling outliers
        sns.boxplot(data=data_cleaned, x=col, ax=axes[1, i])
        axes[1, i].set_title(f'After: {col}')
        axes[1, i].set_xlabel('')

    # Adjust layout to prevent overlap and set tight layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)



st.header(" Step 2: Exploratory Data Analysis (EDA)")
# Step 2: Exploratory Data Analysis (EDA)
with st.expander("EDA"):
    # Dynamically select continuous columns (numerical columns excluding categorical ones)
    continuous_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the statistical summary for continuous variables
    statistical_summary = combined_df[continuous_columns].describe().T

    # Adding additional statistical measures (if needed)
    statistical_summary['skew'] = combined_df[continuous_columns].skew()
    statistical_summary['kurtosis'] = combined_df[continuous_columns].kurtosis()

    # Rename the columns for clarity
    statistical_summary = statistical_summary.rename(columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "25%": "25th Percentile",
        "50%": "Median",
        "75%": "75th Percentile",
        "max": "Maximum",
        "skew": "Skewness",
        "kurtosis": "Kurtosis"
    })

    # Display the summary as a table in Streamlit
    st.write("### Statistical Summary of Continuous Variables:")
    st.dataframe(statistical_summary)




# Step 2.1: Heatmap Plot to Visualize Correlations
with st.expander("Step 2.1: Heatmap Plot to Visualize Correlations"):
    # Compute the correlation matrix for continuous variables
    continuous_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = combined_df[continuous_columns].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Create the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Show correlation values in the heatmap
        fmt=".2f",   # Format for annotation
        cmap="coolwarm",  # Color palette
        linewidths=0.5,  # Line width between cells
        cbar=True,  # Show color bar
    )

    # Add title to the heatmap
    plt.title('Correlation Heatmap of Continuous Variables', fontsize=16)

    # Display the heatmap in Streamlit
    st.pyplot(plt)




# Step 2.2: Heatmap Plot for Each Country
with st.expander("Step 2.2: Heatmap Plot for Each Country"):
    # Get the unique countries
    countries = combined_df['Country'].unique()
    
    # Set up the figure to display the heatmaps horizontally
    fig, axes = plt.subplots(nrows=1, ncols=len(countries), figsize=(15, 8))

    # If only one country, axes might not be an array, so adjust for that
    if len(countries) == 1:
        axes = [axes]
    
    # Loop through each country to generate a heatmap
    for i, country in enumerate(countries):
        # Filter data for the specific country
        country_data = combined_df[combined_df['Country'] == country]
        
        # Compute the correlation matrix for continuous variables
        continuous_columns = country_data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = country_data[continuous_columns].corr()

        # Create the heatmap for the current country
        sns.heatmap(
            correlation_matrix,
            annot=True,  # Show correlation values in the heatmap
            fmt=".2f",   # Format for annotation
            cmap="coolwarm",  # Color palette
            linewidths=0.5,  # Line width between cells
            cbar=True,  # Show color bar
            ax=axes[i]  # Assign heatmap to the corresponding axis
        )

        # Add title for each country's heatmap
        axes[i].set_title(f'Correlation Heatmap: {country}', fontsize=14)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the heatmaps in Streamlit
    st.pyplot(fig)


features = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS', 'WSgust', 'WSstdev']
X = combined_df[features].dropna()  # Drop rows with missing values

# Normalize the data (optional but recommended for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # You can adjust n_clusters based on your data
combined_df['cluster'] = kmeans.fit_predict(X_scaled)

# Streamlit layout with expander for clustering steps
with st.expander("Scatterplot of GHI vs DNI with Clusters"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=combined_df, x='GHI', y='DNI', hue='cluster', palette='viridis', ax=ax)
    ax.set_title("Clustered Data by GHI and DNI")
    st.pyplot(fig)

    # Display cluster centers
    st.write("### Cluster Centers:")
    st.write(kmeans.cluster_centers_)




st.header("Time Series Data Analysis")
# Step 1: Preprocess the Time Series Data
with st.expander("Step 1: Preprocess the Time Series Data"):
    # Ensure 'Timestamp' is in datetime format
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])

    # Set 'Timestamp' as the index
    combined_df.set_index('Timestamp', inplace=True)

    # Show the first few rows of the data
    st.write(combined_df.head())


# Step 2: Extract Time Features
with st.expander("Step 2: Extract Time Features"):
    # Extract additional time-based features for analysis
    combined_df['year'] = combined_df.index.year
    combined_df['month'] = combined_df.index.month
    combined_df['day'] = combined_df.index.day
    combined_df['hour'] = combined_df.index.hour
    combined_df['weekday'] = combined_df.index.weekday  # Monday=0, Sunday=6

    # Show the new columns
    st.write(combined_df[['year', 'month', 'day', 'hour', 'weekday']].head())


# Step 3: Aggregate the Data to Identify Trends
# with st.expander("Step 3: Aggregate the Data to Identify Trends"):
#     # Group by year and month to find the seasonal pattern
#     monthly_avg = combined_df.groupby(['year', 'month'])[['GHI', 'DNI', 'DHI']].mean()

#     # Plotting the monthly trend for GHI
#     st.subheader("Monthly Average GHI")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     monthly_avg['GHI'].plot(ax=ax, title="Monthly Average GHI")
#     st.pyplot(fig)

#     # Group by hour of the day to observe daily patterns
#     hourly_avg = combined_df.groupby('hour')[['GHI', 'DNI', 'DHI']].mean()

#     # Plotting the daily trend for GHI
#     st.subheader("Average GHI by Hour")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     hourly_avg['GHI'].plot(ax=ax, title="Average GHI by Hour")
#     st.pyplot(fig)







# Expander for Step 4: Seasonal Decomposition of Time Series (Trend, Seasonal, and Residual)
# with st.expander("Step 4: Seasonal Decomposition of Time Series (Trend, Seasonal, and Residual)"):
#     # Decompose the time series for GHI (Global Horizontal Irradiance)
#     decomposition = sm.tsa.seasonal_decompose(combined_df['GHI'], model='additive', period=365)  # period=365 for daily data

#     # Plot each component individually
#     fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    
#     # Plot the observed data
#     axs[0].plot(decomposition.observed)
#     axs[0].set_title("Observed Data")
    
#     # Plot the trend component
#     axs[1].plot(decomposition.trend)
#     axs[1].set_title("Trend")
    
#     # Plot the seasonal component
#     axs[2].plot(decomposition.seasonal)
#     axs[2].set_title("Seasonal")
    
#     # Plot the residual component
#     axs[3].plot(decomposition.resid)
#     axs[3].set_title("Residual")

#     st.pyplot(fig)


# Expander for Step 5: Examine Variability in Solar Potential Over Time
# with st.expander("Step 5: Examine Variability in Solar Potential Over Time"):
#     # Calculate 7-day rolling average for GHI to observe trends
#     combined_df['GHI_rolling_avg'] = combined_df['GHI'].rolling(window=7).mean()

#     # Calculate 7-day rolling standard deviation to observe variability
#     combined_df['GHI_rolling_std'] = combined_df['GHI'].rolling(window=7).std()

#     # Plot GHI, Rolling Average, and Rolling Std
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(combined_df['GHI'], label='GHI', color='blue', alpha=0.5)
#     ax.plot(combined_df['GHI_rolling_avg'], label='7-day Rolling Average', color='orange')
#     ax.plot(combined_df['GHI_rolling_std'], label='7-day Rolling Std', color='green')
#     ax.set_title('GHI with Rolling Average and Standard Deviation')
#     ax.legend()
#     st.pyplot(fig)

# Expander for Step 6: Long-Term Variability (Annual Trends)
# with st.expander("Step 6: Long-Term Variability (Annual Trends)"):
#     # Calculate yearly averages for GHI, DNI, etc.
#     yearly_avg = combined_df.groupby('year')[['GHI', 'DNI', 'DHI']].mean()

#     # Plot the average GHI by year
#     fig, ax = plt.subplots(figsize=(12, 6))
#     yearly_avg['GHI'].plot(ax=ax, title="Yearly Average GHI")
#     st.pyplot(fig)

# # Expander for Step 7: Autocorrelation (ACF and PACF)
# with st.expander("Step 7: Autocorrelation (ACF and PACF)"):
#     # Plot ACF and PACF for GHI to check for temporal dependencies
#     fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    
#     plot_acf(combined_df['GHI'].dropna(), lags=50, ax=ax[0])
#     plot_pacf(combined_df['GHI'].dropna(), lags=50, ax=ax[1])
    
#     st.pyplot(fig)
