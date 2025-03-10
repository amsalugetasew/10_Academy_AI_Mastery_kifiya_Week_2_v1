import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
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
    
    def determine_optimal_k(normalized_data, k_range=(2, 10)):
        """
        Determines the optimal number of clusters (k) using the Elbow Method and Silhouette Score.
        
        Parameters:
        normalized_data (DataFrame or array-like): The normalized dataset for clustering.
        k_range (tuple): Range of k values to evaluate (default: (2, 10)).
        
        Returns:
        None (Displays plots for inertia and silhouette scores)
        """
        inertia = []
        silhouette_scores = []
        K = range(k_range[0], k_range[1])
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_data)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(normalized_data, kmeans.labels_))
        
        # Plot the elbow curve
        plt.figure(figsize=(10, 5))
        plt.plot(K, inertia, marker='o', label='Inertia')
        plt.title('Elbow Method to Determine Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.show()
        
        # Plot the silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(K, silhouette_scores, marker='o', color='orange', label='Silhouette Score')
        plt.title('Silhouette Scores to Evaluate k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.show()
    


    def get_top_handset_types(self, handset_metrics, cols, top_n=30):
        """
        Sort and filter for top handset types by average throughput.
        
        Parameters:
            handset_metrics (pd.DataFrame): DataFrame containing aggregated handset metrics.
            top_n (int): Number of top records to select.
        
        Returns:
            pd.DataFrame: DataFrame with top handset types based on average throughput.
        """
        return handset_metrics.sort_values(by=cols, ascending=False).head(top_n)

    def plot_top_handset_throughput(self, top_handset_df):
        """
        Plot the distribution of average throughput per handset type (Top N) as a horizontal bar plot.
        
        Parameters:
            top_handset_df (pd.DataFrame): DataFrame containing top handset types with 'Average Throughput'
                                        and 'Most Common Handset Type' columns.
        """
        plt.figure(figsize=(14, 12))
        ax = sns.barplot(
            data=top_handset_df,
            x='Average Throughput',
            y='Most Common Handset Type',
            color='skyblue'
        )
        plt.title('Top 30 Handset Types by Average Throughput')
        plt.xlabel('Average Throughput (Mbps)')
        plt.ylabel('Handset Type')

        # Add value annotations to each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}', 
                        (p.get_width() + 0.5, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()



    def plot_tcp_retransmission(self, top_retransmission):
        """
        Plot average TCP retransmission volume (DL and UL) per handset type as a horizontal bar plot.
        
        Parameters:
            top_retransmission (pd.DataFrame): DataFrame containing top handset types with the following columns:
                - 'Most Common Handset Type'
                - 'Avg TCP DL Retransmission'
                - 'Avg TCP UL Retransmission'
        """
        plt.figure(figsize=(14, 12))
        
        # Plot DL retransmission bars
        ax = sns.barplot(
            data=top_retransmission,
            x='Avg TCP DL Retransmission',
            y='Most Common Handset Type',
            color='salmon', label='DL Retransmission'
        )
        
        # Plot UL retransmission bars over the same y-axis
        sns.barplot(
            data=top_retransmission,
            x='Avg TCP UL Retransmission',
            y='Most Common Handset Type',
            color='blue', alpha=0.7, label='UL Retransmission'
        )
        
        # Annotate DL (and UL) retransmission values on the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}', 
                        (p.get_width() + 0.5, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=10, color='black')
        
        plt.title('Top Handset Types by Average TCP Retransmission Volume')
        plt.xlabel('TCP Retransmission Volume (Bytes)')
        plt.ylabel('Handset Type')
        plt.legend()
        plt.tight_layout()
        plt.show()



    


    def clean_handset_data(self, df):
        """
        Clean the 'Most Common Handset Type' column in the DataFrame.
        Replace 'undefined' values with 'Unknown'.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.copy()
        df['Most Common Handset Type'] = df['Most Common Handset Type'].replace('undefined', 'Unknown')
        return df

    def standardize_features(self, df, features):
        """
        Standardize the selected features using StandardScaler.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the features.
            features (list): List of feature column names to standardize.
            
        Returns:
            np.ndarray: Scaled data.
            StandardScaler: Fitted scaler.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        return scaled_data, scaler

    def perform_kmeans_clustering(self, scaled_data, n_clusters=3, random_state=42):
        """
        Apply K-Means clustering on the scaled data.
        
        Parameters:
            scaled_data (np.ndarray): The standardized feature data.
            n_clusters (int): Number of clusters.
            random_state (int): Random state for reproducibility.
            
        Returns:
            KMeans: Fitted KMeans model.
            np.ndarray: Cluster labels for each sample.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(scaled_data)
        return kmeans, labels

    def plot_clusters(self, df, x_feature, y_feature, cluster_col='Cluster Label'):
        """
        Plot clusters using a scatter plot for two selected features.
        
        Parameters:
            df (pd.DataFrame): DataFrame that includes cluster labels.
            x_feature (str): Column name for the x-axis.
            y_feature (str): Column name for the y-axis.
            cluster_col (str): Column name containing cluster labels.
        """
        plt.figure(figsize=(8, 6))
        clusters = df[cluster_col].unique()
        
        for cluster in clusters:
            subset = df[df[cluster_col] == cluster]
            plt.scatter(
                subset[x_feature],
                subset[y_feature],
                label=cluster
            )
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title('K-Means Clustering of Users')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

    def interpret_clusters(self, df, features, cluster_col='Cluster'):
        """
        Compute average metrics for each cluster to help interpret the clusters.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the original features and cluster labels.
            features (list): List of feature column names to compute averages for.
            cluster_col (str): Column name containing cluster labels.
            
        Returns:
            pd.DataFrame: DataFrame containing average metrics per cluster.
        """
        cluster_descriptions = []
        clusters = df[cluster_col].unique()
        
        for cluster in clusters:
            cluster_data = df[df[cluster_col] == cluster]
            description = {'Cluster': cluster}
            for feature in features:
                description[f'Avg {feature}'] = cluster_data[feature].mean()
            cluster_descriptions.append(description)
        
        return pd.DataFrame(cluster_descriptions)
    


    def compute_aggregated_metrics(self, df):
        """
        Compute aggregated metrics:
        - 'Average Throughput' as the mean of DL and UL throughput.
        - 'Average TCP Retransmission' as the mean of DL and UL TCP retransmission volumes.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the required columns.
            
        Returns:
            pd.DataFrame: DataFrame with new aggregated columns.
        """
        df = df.copy()

        df['Average Throughput'] = (df['Average DL Throughput'] + df['Average UL Throughput']) / 2
        df['Average TCP Retransmission'] = (df['Avg TCP DL Retransmission'] + df['Avg TCP UL Retransmission']) / 2
        return df

    def normalize_features(self, df, features):
        """
        Normalize selected features using StandardScaler.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing features.
            features (list): List of features to normalize.
            
        Returns:
            normalized_data (np.ndarray): Normalized feature array.
            scaler (StandardScaler): Fitted scaler object.
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[features])
        return normalized_data, scaler

    def perform_clustering(self, normalized_data, n_clusters=3, random_state=42):
        """
        Perform K-Means clustering on normalized data.
        
        Parameters:
            normalized_data (np.ndarray): Array of normalized features.
            n_clusters (int): Number of clusters.
            random_state (int): Seed for reproducibility.
            
        Returns:
            kmeans (KMeans): Fitted KMeans model.
            clusters (np.ndarray): Cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(normalized_data)
        return kmeans, clusters

    def compute_cluster_summary(self, df, cluster_col='Cluster'):
        """
        Compute average, minimum, and maximum for key metrics per cluster.
        
        Parameters:
            df (pd.DataFrame): DataFrame with cluster labels and aggregated metrics.
            cluster_col (str): Column name for clusters.
            
        Returns:
            pd.DataFrame: Cluster summary statistics.
        """
        cluster_summary = df.groupby(cluster_col).agg({
            'Average Throughput': ['mean', 'min', 'max'],
            'Average TCP Retransmission': ['mean', 'min', 'max']
        }).reset_index()
        
        # Rename columns for clarity
        cluster_summary.columns = ['Cluster', 
                                'Avg Throughput Mean', 'Avg Throughput Min', 'Avg Throughput Max', 
                                'Avg TCP Retr Mean', 'Avg TCP Retr Min', 'Avg TCP Retr Max']
        return cluster_summary

    def plot_clusters1(self, df, x_feature='Average Throughput', y_feature='Average TCP Retransmission', cluster_col='Cluster Label'):
        """
        Visualize clusters using a scatter plot.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the metrics and cluster labels.
            x_feature (str): Feature for the x-axis.
            y_feature (str): Feature for the y-axis.
            cluster_col (str): Column name with cluster labels.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=x_feature, 
            y=y_feature, 
            hue=cluster_col, 
            data=df, 
            palette='Set1'
        )
        plt.title('K-Means Clustering of User Experience')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()
    

    def plot_actual_vs_predicted(self, df, actual_col='Actual Satisfaction Score', predicted_col='Predicted Satisfaction Score'):
        """
        Visualize the relationship between actual and predicted satisfaction scores.

        Parameters:
            df (pd.DataFrame): DataFrame containing the actual and predicted scores.
            actual_col (str): Column name for actual satisfaction scores.
            predicted_col (str): Column name for predicted satisfaction scores.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual_col, y=predicted_col, data=df, color='blue', label='Predicted vs Actual')

        # Plot the perfect prediction line
        min_value = min(df[actual_col].min(), df[predicted_col].min())
        max_value = max(df[actual_col].max(), df[predicted_col].max())
        plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Perfect Prediction')

        plt.xlabel('Actual Satisfaction Score')
        plt.ylabel('Predicted Satisfaction Score')
        plt.title('Actual vs. Predicted Satisfaction Scores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    def plot_residuals(self, df, actual_col='Actual Satisfaction Score', predicted_col='Predicted Satisfaction Score'):
        """
        Visualize the residuals of the model's predictions.

        Parameters:
            df (pd.DataFrame): DataFrame containing the actual and predicted scores.
            actual_col (str): Column name for actual satisfaction scores.
            predicted_col (str): Column name for predicted satisfaction scores.
        """
        residuals = df[actual_col] - df[predicted_col]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=predicted_col, y=residuals, data=df, color='green', label='Residuals')

        plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual')
        plt.xlabel('Predicted Satisfaction Score')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted Satisfaction Scores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def visualize_clusters(self, df):
        # Set the plot size
        plt.figure(figsize=(10, 6))

        # Create a scatter plot of Engagement Score vs. Experience Score
        sns.scatterplot(data=df, x='Engagement Score', y='Experience Score', hue='Cluster Name', palette='viridis', s=100)

        # Add plot labels and title
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.title('Customer Satisfaction Clusters')
        plt.legend(title='Cluster Name')
        plt.grid(True)
        plt.show()
    

    def visualize_clusters(self, df):
        # Set the plot size
        plt.figure(figsize=(15, 10))

        # Create a 3D axis
        ax = plt.axes(projection='3d')

        # Define a color palette with distinct colors for each cluster
        palette = sns.color_palette('viridis', as_cmap=True)
        cluster_labels = df['Cluster Name'].unique()
        colors = [palette(i / len(cluster_labels)) for i in range(len(cluster_labels))]

        # Scatter plot for each cluster
        for i, cluster in enumerate(cluster_labels):
            cluster_data = df[df['Cluster Name'] == cluster]
            ax.scatter3D(cluster_data['Engagement Score'],
                        cluster_data['Experience Score'],
                        cluster_data['Satisfaction Score'],
                        color=colors[i],
                        label=cluster,
                        s=100)

        # Add plot labels and title
        ax.set_xlabel('Engagement Score')
        ax.set_ylabel('Experience Score')
        ax.set_zlabel('Satisfaction Score')
        ax.set_title('3D Visualization of Customer Satisfaction Clusters')

        # Add a legend
        ax.legend(title='Cluster Name')

        # Show the plot
        plt.show()

    
    def correlation_matrix1(self,df):
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.2f')
        plt.title('Correlation Heatmap of Scores')
        plt.show()