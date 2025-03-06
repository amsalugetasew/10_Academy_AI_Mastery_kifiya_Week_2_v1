import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
class Plot:
    def __init__(self, dataframe):
        """
        Initialize the Plot class with a dataframe.

        Parameters:
        dataframe (pd.DataFrame): The dataframe to analyze and plot.
        """
        self.dataframe = dataframe

    def distribution_of_missing_values(self, column_names):
        """
        Plot the distribution of missing values for the specified columns.

        Parameters:
        column_names (list): List of column names to check for missing values.
        """
        # Check for missing values in the specified columns
        null_columns = self.dataframe[column_names]
        constant_values = (null_columns.isnull().sum())

        # Calculate percentages
        total = constant_values.sum()
        percentages = (constant_values / total) * 100 if total > 0 else [0] * len(constant_values)

        # Bar plot for constant/zero values
        ax = constant_values.plot(kind='barh', figsize=(10, 8), color='orange')

        # Add number and percentage annotations
        for bar, percentage in zip(ax.patches, percentages):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            annotation = f'{int(width)} ({percentage:.2f}%)'
            ax.annotate(annotation, xy=(width, y), xytext=(5, 0),
                        textcoords="offset points", ha='left', va='center')

        # Add labels and title
        plt.title("Distribution of Missing Values Per Column")
        plt.xlabel("Frequency")
        plt.ylabel("Columns")
        plt.xticks(rotation=45)
        plt.show()
    
    
    def plot_quantitative_data(self, quantitative_vars, style="whitegrid"):
        """
        Plot histograms and KDE for the specified quantitative variables.

        Parameters:
        quantitative_vars (list): List of quantitative variable column names.
        style (str): Seaborn style for the plots (default is "whitegrid").
        """
        # Set the style for seaborn plots
        sns.set(style=style)

        # Create a figure with subplots
        num_vars = len(quantitative_vars)
        rows = num_vars // 2 + num_vars % 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 12))

        # Flatten axes array for easier iteration
        axes = axes.flatten()

        # Plot each variable in a subplot
        for i, var in enumerate(quantitative_vars):
            ax = axes[i]  # Get the current subplot position
            
            # Histogram and KDE
            sns.histplot(self.dataframe[var], bins=30, kde=True, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram and KDE of {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Frequency')

        # Adjust layout for readability
        plt.tight_layout()
        plt.show()
        
    
    def plot_boxplots(self, quantitative_vars, style="whitegrid"):
        """
        Plot boxplots for the specified quantitative variables.

        Parameters:
        quantitative_vars (list): List of quantitative variable column names.
        style (str): Seaborn style for the plots (default is "whitegrid").
        """
        # Set the style for seaborn plots
        sns.set(style=style)

        # Create a figure with subplots
        num_vars = len(quantitative_vars)
        rows = num_vars // 2 + num_vars % 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 12))

        # Flatten axes array for easier iteration
        axes = axes.flatten()

        # Plot each variable in a subplot
        for i, var in enumerate(quantitative_vars):
            ax = axes[i]  # Get the current subplot position
            
            # Box plot
            sns.boxplot(data=self.dataframe, y=var, ax=ax, color='lightcoral')
            ax.set_title(f'Box Plot of {var}')
            ax.set_xlabel('')  # No x-axis label needed for box plot
            ax.set_ylabel(var)

        # Adjust layout for readability
        plt.tight_layout()
        plt.show()
    
    
    
    def plot_scatter_with_total_dlul(self):
        """
        Create a new column for total data (Total DL + UL) and plot scatter plots
        of each application vs. the total data.

        This method doesn't require passing parameters.
        """
        # Create a new column for the total data (DL + UL)
        self.dataframe['Total DL+UL (Bytes)'] = (
            self.dataframe['Total Youtube (Bytes)'] + 
            self.dataframe['Total Netflix (Bytes)'] + 
            self.dataframe['Total Gaming (Bytes)'] + 
            self.dataframe['Total Other (Bytes)']
        )

        # List of applications to explore
        applications = ['Total Youtube (Bytes)', 'Total Netflix (Bytes)', 
                        'Total Gaming (Bytes)', 'Total Other (Bytes)']

        # Create a figure to hold scatter plots for each application vs Total DL+UL
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Loop through each application and plot scatter plots
        for i, app in enumerate(applications):
            ax = axes[i // 2, i % 2]  # Determine subplot position

            # Scatter plot of the relationship between application data and total data
            sns.scatterplot(x=self.dataframe[app], y=self.dataframe['Total DL+UL (Bytes)'], ax=ax)
            ax.set_title(f'Relationship between {app} and Total DL+UL')
            ax.set_xlabel(app)
            ax.set_ylabel('Total DL+UL (Bytes)')

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
    
    
    def plot_correlation_heatmap(self, variables, title, cmap='RdYlGn'):
        """
        Plot a correlation heatmap for the specified variables.

        Parameters:
        variables (list): List of variables to compute the correlation matrix for.
        title (str): Title of the heatmap plot.
        cmap (str): Color map for the heatmap (default is 'RdYlGn').
        """
        # Compute the correlation matrix
        correlation_matrix = self.dataframe[variables].corr()

        # Set the figure size for the heatmap
        plt.figure(figsize=(14, 10))  # You can adjust the size here

        # Create the heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.2f', 
                    linewidths=0.5, cbar_kws={'shrink': 0.75})  # Adjust the color bar size

        # Set the title of the plot
        plt.title(title, fontsize=16)

        # Show the plot
        plt.show()
    
    
    def perform_pca(self, columns_to_pca, n_components=2):
        """
        Perform PCA on selected columns, standardize the data, and reduce it to a specified number of components.

        Parameters:
        columns_to_pca (list): List of columns to be included in PCA.
        n_components (int): Number of principal components to keep (default is 2).
        """
        # Step 1: Standardize the data
        scaler = StandardScaler()
        data_scaled = self.dataframe[columns_to_pca]
        data_scaled['Total Social Media (Bytes)'] = data_scaled['Social Media DL (Bytes)'] + data_scaled['Social Media UL (Bytes)']
        data_scaled['Total Google (Bytes)'] = data_scaled['Google DL (Bytes)'] + data_scaled['Google UL (Bytes)']
        data_scaled['Total Email (Bytes)'] = data_scaled['Email DL (Bytes)'] + data_scaled['Email UL (Bytes)']
        data_scaled['Total Youtube (Bytes)'] = data_scaled['Youtube DL (Bytes)'] + data_scaled['Youtube UL (Bytes)']
        data_scaled['Total Netflix (Bytes)'] = data_scaled['Netflix DL (Bytes)'] + data_scaled['Netflix UL (Bytes)']
        data_scaled['Total Gaming (Bytes)'] = data_scaled['Gaming DL (Bytes)'] + data_scaled['Gaming UL (Bytes)']
        data_scaled['Total Other (Bytes)'] = data_scaled['Other DL (Bytes)'] + data_scaled['Other UL (Bytes)']
        columns_to_pca = ['Total Social Media (Bytes)', 'Total Google (Bytes)', 'Total Email (Bytes)','Total Youtube (Bytes)',
                          'Total Netflix (Bytes)','Total Gaming (Bytes)', 'Total Other (Bytes)']
        data_scaled_f = scaler.fit_transform(data_scaled[columns_to_pca])

        # Step 2: Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(data_scaled_f)

        # Step 3: Explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Step 4: Calculate the cumulative explained variance
        cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

        # Step 5: Print the results
        print("Explained Variance Ratio for each component:")
        print(explained_variance_ratio)

        print("\nCumulative Explained Variance:")
        print(cumulative_explained_variance)

        # Step 6: Reduced data (2D if n_components=2)
        reduced_data = pca.transform(data_scaled_f)

        print(f"\nFirst {n_components} principal components:")
        print(reduced_data[:5])  # Show the first 5 rows of the reduced data

        # Optional: If you want to plot the first two principal components
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, c='orange')
            plt.title('PCA - First 2 Principal Components')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()
    
    
    def plot_application_traffic(self, app_traffic):
        """
        Aggregate and plot the total traffic per application.

        Parameters:
        app_traffic (pd.DataFrame): DataFrame containing traffic information for different applications.
        """
        app_traffic = self.dataframe
        # Aggregate total traffic per application
        application_totals = {
            'Youtube': app_traffic['Youtube Traffic'].sum(),
            'Netflix': app_traffic['Netflix Traffic'].sum(),
            'Gaming': app_traffic['Gaming Traffic'].sum()
        }

        # Plot the data
        plt.figure(figsize=(8, 6))
        plt.bar(application_totals.keys(), application_totals.values(), color=['red', 'blue', 'green'])
        plt.title('Top 3 Most Used Applications by Total Traffic')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xlabel('Application')
        plt.show()
        
    
    
    def plot_kmeans_evaluation(self, dataframe, columns_to_cluster, k_range=range(2, 10)):
        """
        Evaluate KMeans clustering using the Elbow Method and Silhouette Scores.

        Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data to cluster.
        columns_to_cluster (list): List of column names to include in clustering.
        k_range (range): Range of k values to test (default is range(2, 10)).
        """
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(dataframe[columns_to_cluster])
        
        inertia = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))
        
        # Plot the elbow curve
        plt.figure(figsize=(10, 5))
        plt.plot(k_range, inertia, marker='o', label='Inertia')
        plt.title('Elbow Method to Determine Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()
        
        # Plot the silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(k_range, silhouette_scores, marker='o', color='orange', label='Silhouette Score')
        plt.title('Silhouette Scores to Evaluate k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.show()

    
    def plot_lower_triangle_correlation(self, df, start_col, end_col, title):
        """
        Plots a lower triangle heatmap of the correlation matrix for the selected numeric columns.

        Parameters:
        df (DataFrame): The input dataframe.
        start_col (int): The starting column index.
        end_col (int): The ending column index.
        title (str): The title of the heatmap.
        """
        # Select the numeric subset
        subset = df.select_dtypes(include=['number']).iloc[:, start_col:end_col]

        # Compute correlation matrix
        corr_matrix = subset.corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn")
        plt.title(title)
        plt.show()

    def analyze_correlation(self, df):
        """
        Analyzes the correlation of numeric columns in the dataframe and plots the lower triangle correlation matrices.

        Parameters:
        df (DataFrame): The input dataframe.
        """
        # Select only numeric columns
        numeric_data = df.select_dtypes(include=['number'])
        num_columns = len(numeric_data.columns)

        if num_columns < 20:
            print("The dataset has fewer than 20 numeric columns.")
            return

        # Plot first 20 numeric columns correlation matrix
        self.plot_lower_triangle_correlation(df, start_col=3, end_col=27, 
                                        title="Lower Triangle Correlation Matrix (First 20 Numeric Columns)")

        if num_columns > 27:
            # Plot second 20 numeric columns correlation matrix
            self.plot_lower_triangle_correlation(df, start_col=27, end_col=55, 
                                            title="Lower Triangle Correlation Matrix (Second 20 Numeric Columns)")
        else:
            print("The dataset has fewer than 40 numeric columns.")
    def plot_missing_values_heatmap(self,df):
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
    
    def plot_histograms_with_lines(self, group, title):
        """
        Plots histograms with vertical lines for mean and median.
        
        Parameters:
        group (DataFrame): A subset of numeric columns from the dataset.
        title (str): Title for the histograms.
        """
        for column in group.columns:
            plt.figure(figsize=(8, 5))
            group[column].hist(bins=30, color='skyblue', edgecolor='black', alpha=0.7)

            # Add vertical lines for mean and median
            mean_val = group[column].mean()
            median_val = group[column].median()

            plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f}')

            # Add title and legend
            plt.title(f'Histogram of {column}\n{title}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

    def analyze_numeric_columns(self, df):
        """
        Analyzes the numeric columns in the dataset by:
        - Checking skewness
        - Plotting histograms with mean and median lines
        
        Parameters:
        df (DataFrame): The input dataset.
        """
        # Select numeric columns
        numeric_data = df.select_dtypes(include=['number'])
        num_columns = len(numeric_data.columns)

        if num_columns < 10:
            print("The dataset has fewer than 10 numeric columns.")
            return
        
        # Define groups of numeric columns
        column_groups = [
            (3, 12, "First Group of Numeric Columns"),
            (12, 24, "Second Group of Numeric Columns"),
            (24, 40, "Third Group of Numeric Columns"),
            (40, 55, "Fourth Group of Numeric Columns")
        ]

        for start, end, title in column_groups:
            if num_columns > start:
                group = numeric_data.iloc[:, start:end]
                skewness = group.skew()
                print(f"Skewness for {title}:")
                print(skewness)
                self.plot_histograms_with_lines(group, title)
            else:
                print(f"The dataset has fewer than {end} numeric columns.")
                break  # Stop checking further groups if columns are insufficient

    
    
    def quantile_impute_based_on_skewness(self, df, col):
        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Skipping column '{col}' because it is not numeric.")
            return
        
        skewness = df[col].skew()  # Calculate skewness of the column
        
        # If the skewness is greater than 0 (right-skewed), use a higher quantile (e.g., 75th percentile)
        if skewness > 0:
            quantile_value = df[col].quantile(0.75)  # 75th percentile for right-skewed data
            df[col].fillna(quantile_value, inplace=True)
        
        # If the skewness is less than 0 (left-skewed), use a lower quantile (e.g., 25th percentile)
        elif skewness < 0:
            quantile_value = df[col].quantile(0.25)  # 25th percentile for left-skewed data
            df[col].fillna(quantile_value, inplace=True)
        
        # If the skewness is approximately 0 (symmetrical), you can choose a quantile around the median (50th percentile)
        else:
            print(f"Approximately symmetric: {col}, applying Quantile Imputation (50th percentile).")
            quantile_value = df[col].quantile(0.50)  # 50th percentile for symmetric data (similar to median)
            df[col].fillna(quantile_value, inplace=True)