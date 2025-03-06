import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class TelecomUserAnalysis:
    def __init__(self, df):
        """Initialize with telecom data."""
        self.df = df

    # --------------------------- 1️⃣ USER OVERVIEW ANALYSIS ---------------------------
    def user_overview(self):
        """Perform user overview analysis, including statistics, segmentation, and trends."""
        print("\n📊 USER OVERVIEW ANALYSIS")
        
        # 1.1 Basic statistics
        print("\n🔹 Basic Statistics:")
        print(self.df.describe())

        # 1.2 Distribution of Data Usage
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df["Total DL (Bytes)"] / 1e6, bins=30, kde=True, color='blue')
        plt.xlabel("Total Download (MB)")
        plt.title("Data Usage Distribution")
        plt.show()

        # 1.3 Customer Segmentation using K-Means Clustering
        X = self.df[["Total UL (Bytes)", "Total DL (Bytes)", "Duration (seconds)"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df["Cluster"] = kmeans.fit_predict(X_scaled)
        print("\n🔹 Customer Segments Assigned (0, 1, 2):")
        print(self.df["Cluster"].value_counts())

        # 1.4 Calculate Customer Lifetime Value (CLV)
        self.df["CLV"] = (self.df["Total DL (Bytes)"] + self.df["Total UL (Bytes)"]) / self.df["Duration (seconds)"]
        print("\n🔹 Average Customer Lifetime Value (CLV):")
        print(self.df["CLV"].mean())
    
    def segment_by_duration(self):
        """Segments users based on call duration."""
        print("\n📊 **Duration-Based Segmentation**")

        bins = [0, 60, 300, self.df["Duration (seconds)"].max()]
        labels = ["low duration (<1 min)", "Medium duration (1-5 min)", "Long duration (>5 min)"]
        self.df["Duration_Segment"] = pd.cut(self.df["Duration (seconds)"], bins=bins, labels=labels)

        print(self.df["Duration_Segment"].value_counts())

        # Visualization
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="Duration_Segment", palette="coolwarm")
        plt.title("Call Duration Segmentation")
        plt.xlabel("Call Duration Category")
        plt.ylabel("Number of Users")
        plt.show()
    
    def segment_by_location(self, st):
        """Segments users based on their last recorded location."""
        print("\n📊 **Location-Based Segmentation**")
        if st == 'top':
            if "Last Location Name" in self.df.columns:
                location_counts = self.df["Last Location Name"].value_counts().head(10)
                print("\n🔹 **Top 10 Locations**:\n", location_counts)

                # Visualization
                plt.figure(figsize=(12, 6))
                sns.barplot(x=location_counts.index, y=location_counts.values, palette="viridis")
                plt.xticks(rotation=45)
                plt.title("Top 10 User Locations")
                plt.xlabel("Location")
                plt.ylabel("Number of Users")
                plt.show()
        else:
            if "Last Location Name" in self.df.columns:
                location_counts = self.df["Last Location Name"].value_counts().tail(10)
                print("\n🔹 **Least Engaged 10 Locations**:\n", location_counts)

                # Visualization
                plt.figure(figsize=(12, 6))
                sns.barplot(x=location_counts.index, y=location_counts.values, palette="viridis")
                plt.xticks(rotation=45)
                plt.title("least 10 User Locations")
                plt.xlabel("Location")
                plt.ylabel("Number of Users")
                plt.show()

    def segment_by_usage(self):
        """Segments users based on data usage."""
        print("\n📊 **Usage-Based Segmentation**")

        bins = [0, self.df["Total DL (Bytes)"].quantile(0.33), 
                   self.df["Total DL (Bytes)"].quantile(0.67), 
                   self.df["Total DL (Bytes)"].max()]
        labels = ["Low Data User", "Medium Data User", "High Data User"]
        self.df["Usage_Segment"] = pd.cut(self.df["Total DL (Bytes)"], bins=bins, labels=labels)

        print(self.df["Usage_Segment"].value_counts())

        # Visualization
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="Usage_Segment", palette="muted")
        plt.title("User Data Usage Segmentation")
        plt.xlabel("Data Usage Category")
        plt.ylabel("Number of Users")
        plt.show()

    def multi_metric_segmentation(self):
        """Performs clustering using K-Means on multiple metrics."""
        print("\n📊 **Multi-Metric Segmentation (K-Means Clustering)**")

        features = ["Total UL (Bytes)", "Total DL (Bytes)", "Duration (seconds)"]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(self.df[features])

        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df["Cluster"] = kmeans.fit_predict(df_scaled)

        print("\n🔹 **K-Means Cluster Distribution**:\n", self.df["Cluster"].value_counts())

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df["Total DL (Bytes)"], y=self.df["Duration (seconds)"], hue=self.df["Cluster"], palette="coolwarm")
        plt.xlabel("Total Download (Bytes)")
        plt.ylabel("Usage Duration (Seconds)")
        plt.title("User Segmentation Based on Usage & Duration (K-Means Clustering)")
        plt.legend(title="Cluster")
        plt.show()

    def multi_metric_segmentation(self, n_clusters=3):
        """Performs K-Means clustering on multiple metrics and assigns meaningful labels."""
        print("\n📊 **Multi-Metric Segmentation (K-Means Clustering)**")

        # Features to be used for clustering
        features = ["Total UL (Bytes)", "Total DL (Bytes)", "Duration (seconds)"]

        # Standardize data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(self.df[features])

        # Determine optimal clusters using the Elbow Method (optional)
        if n_clusters is None:
            distortions = []
            K_range = range(1, 10)  # Test cluster numbers from 1 to 10
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(df_scaled)
                distortions.append(km.inertia_)
            
            # Plot Elbow Method graph
            plt.figure(figsize=(8, 5))
            plt.plot(K_range, distortions, marker='o', linestyle='-', color='b')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia (Distortion)")
            plt.title("Elbow Method for Optimal K Selection")
            plt.show()

            # User can manually determine the best number of clusters
            n_clusters = int(input("Enter the optimal number of clusters: "))

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["Cluster"] = kmeans.fit_predict(df_scaled)

        # Define cluster labels
        cluster_labels = {0: "Light Users", 1: "Moderate Users", 2: "Heavy Users"}
        self.df["Cluster Label"] = self.df["Cluster"].map(cluster_labels)

        print("\n🔹 **K-Means Cluster Distribution**:\n", self.df["Cluster Label"].value_counts())

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df["Total DL (Bytes)"], 
                        y=self.df["Duration (seconds)"], 
                        hue=self.df["Cluster Label"], 
                        palette="coolwarm", 
                        s=50, edgecolor="black")

        plt.xlabel("Total Download (Bytes)")
        plt.ylabel("Usage Duration (Seconds)")
        plt.title("User Segmentation Based on Usage & Duration (K-Means Clustering)")
        plt.legend(title="Cluster Type")
        plt.grid(True)
        plt.show()



    # --------------------------- 2️⃣ USER ENGAGEMENT ANALYSIS ---------------------------
    def user_engagement(self):
        """Analyze user engagement based on activity metrics and churn prediction."""
        print("\n📊 USER ENGAGEMENT ANALYSIS")
        
        # 2.1 Active vs. Inactive Users (Threshold-Based)
        self.df["Active"] = np.where(self.df["Total DL (Bytes)"] > self.df["Total DL (Bytes)"].median(), 1, 0)
        print("\n🔹 Active Users Count:")
        print(self.df["Active"].value_counts())

        # 2.2 Engagement Trends Over Time
        self.df["Start"] = pd.to_datetime(self.df["Start"])
        daily_engagement = self.df.groupby(self.df["Start"].dt.date)["Total DL (Bytes)"].sum()
        
        plt.figure(figsize=(12, 5))
        plt.plot(daily_engagement, marker="o", linestyle="-", color="green")
        plt.xlabel("Date")
        plt.ylabel("Total Data Downloaded (Bytes)")
        plt.title("User Engagement Over Time")
        plt.xticks(rotation=45)
        plt.show()

        # 2.3 Churn Prediction (Using Machine Learning)
        features = ["Total UL (Bytes)", "Total DL (Bytes)", "Duration (seconds)"]
        X = self.df[features]
        y = self.df["Active"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\n🔹 Churn Prediction Report:")
        print(classification_report(y_test, y_pred))

    # --------------------------- 3️⃣ USER EXPERIENCE ANALYSIS ---------------------------
    def user_experience(self):
        """Analyze user experience based on service quality metrics."""
        print("\n📊 USER EXPERIENCE ANALYSIS")
        
        # 3.1 Call Duration Insights
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df["Duration (seconds)"] / 60, color="red")
        plt.xlabel("Call Duration (minutes)")
        plt.title("Call Duration Distribution")
        plt.show()

        # 3.2 Identify Potential Network Issues (Short Calls)
        poor_experience_users = self.df[self.df["Duration (seconds)"] < 30]
        print("\n🔹 Users with Short Call Duration (Potential Quality Issues):")
        print(poor_experience_users[["IMSI", "MSISDN/Number", "Duration (seconds)"]].head())

        # 3.3 Network Speed Analysis
        self.df["Data Speed"] = self.df["Total DL (Bytes)"] / self.df["Duration (seconds)"]
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df["Data Speed"], bins=30, kde=True, color="purple")
        plt.xlabel("Download Speed (Bytes/sec)")
        plt.title("Data Speed Distribution")
        plt.show()

    # --------------------------- 4️⃣ USER SATISFACTION ANALYSIS ---------------------------
    def user_satisfaction(self):
        """Analyze customer satisfaction using CSAT and NPS metrics."""
        print("\n📊 USER SATISFACTION ANALYSIS")
        
        # 4.1 Assign Random CSAT Scores (1-5 Scale)
        self.df["CSAT_Score"] = np.random.randint(1, 6, size=len(self.df))
        avg_csat = self.df["CSAT_Score"].mean()
        print(f"\n🔹 Average CSAT Score: {avg_csat:.2f} (Scale 1-5)")

        # 4.2 Net Promoter Score (NPS)
        promoters = (self.df["CSAT_Score"] >= 4).sum()
        detractors = (self.df["CSAT_Score"] <= 2).sum()
        total_responses = len(self.df)

        nps_score = ((promoters - detractors) / total_responses) * 100
        print(f"\n🔹 Net Promoter Score (NPS): {nps_score:.2f}")

        # 4.3 Correlation Between Usage and Satisfaction
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=self.df["Total DL (Bytes)"], y=self.df["CSAT_Score"], color="orange")
        plt.xlabel("Total Download (Bytes)")
        plt.ylabel("CSAT Score")
        plt.title("Data Usage vs. Satisfaction Score")
        plt.show()

    # --------------------------- 5️⃣ RUN ALL ANALYSIS ---------------------------
    def run_all_analysis(self):
        """Execute all analysis functions in sequence."""
        self.user_overview()
        self.user_engagement()
        self.user_experience()
        self.user_satisfaction()


