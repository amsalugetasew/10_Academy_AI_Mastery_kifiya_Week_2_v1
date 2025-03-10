from sklearn.metrics.pairwise import euclidean_distances
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class UserSatisfactionAnalysis:
    def satisfaction_computing(self, df_engagement, df_experience):
        # Identify categorical and numerical features
        categorical_features = df_experience.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df_experience.select_dtypes(include=['number']).columns.tolist()

        # Define preprocessing for numerical and categorical features
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Create a ColumnTransformer to apply appropriate transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply preprocessing to the experience features
        experience_features = df_experience#.drop(columns=['Most Common Handset Type'])
        processed_experience = preprocessor.fit_transform(experience_features)

        # Define the worst experience cluster (example: first row of processed experience data)
        worst_experience_cluster = processed_experience[0].reshape(1, -1)

        # Standardize engagement features
        scaler_engagement = StandardScaler()
        engagement_features = df_engagement.drop(columns=['MSISDN/Number'])
        scaled_engagement = scaler_engagement.fit_transform(engagement_features)

        # Define the less engaged cluster (example: first row of scaled engagement data)
        less_engaged_cluster = scaled_engagement[0].reshape(1, -1)

        # Calculate Engagement Scores
        engagement_scores = euclidean_distances(scaled_engagement, less_engaged_cluster).flatten()

        # Calculate Experience Scores
        experience_scores = euclidean_distances(processed_experience, worst_experience_cluster).flatten()

        # Combine results into a DataFrame
        results = pd.DataFrame({
            'MSISDN/Number': df_engagement['MSISDN/Number'],
            'Engagement Score': engagement_scores,
            'Experience Score': experience_scores
        })
        results.set_index('MSISDN/Number', inplace=True)

        return results
        
    def calculate_top_satisfaction(self, results):
        # Calculate the Satisfaction Score
        results['Satisfaction Score'] = (results['Engagement Score'] + results['Experience Score']) / 2

        # Sort by Satisfaction Score in descending order
        top_satisfied_customers = results.sort_values(by='Satisfaction Score', ascending=False)
        return top_satisfied_customers
    
    
    def train_regression_model(self, results):
        # Prepare features (Engagement Score and Experience Score) and target (Satisfaction Score)
        X = results[['Engagement Score', 'Experience Score']]
        y = results['Satisfaction Score']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the regression model
        model = LinearRegression()

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict the satisfaction scores on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print model evaluation metrics
        print("Model Evaluation:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R²) Score: {r2}")

        # Optionally, print the predicted vs actual values
        predictions_df = pd.DataFrame({
            'Actual Satisfaction Score': y_test,
            'Predicted Satisfaction Score': y_pred
        })
        return predictions_df
    
    
    def perform_kmeans_clustering(self, results):
        # Prepare the data (combining engagement and experience scores)
        X = results[['Satisfaction Score']]

        # Initialize the KMeans model with k=2
        kmeans = KMeans(n_clusters=3, random_state=42)

        # Fit the model to the data
        results['Cluster'] = kmeans.fit_predict(X)
        return results


    def assign_cluster_names(self, df):
        # Define cluster names based on your analysis
        cluster_names = {
            0: 'Low Satisfaction',
            1: 'High Satisfaction',
            2: 'Moderate Satisfaction'
        }
        # Map the cluster labels to descriptive names
        df['Cluster Name'] = df['Cluster'].map(cluster_names)
        return df