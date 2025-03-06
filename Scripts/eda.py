import pandas as pd

class EDA:
    """
    A class to perform basic exploratory data analysis (EDA) on a given pandas DataFrame.
    """

    def __init__(self, data):
        """
        Initialize the EDA class with a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to analyze.
        """
        self.data = data

    def display_top_n(self, n=5):
        """
        Display the top n rows of the DataFrame.

        Args:
            n (int): The number of rows to display. Default is 5.

        Returns:
            pd.DataFrame: The top n rows of the DataFrame.
        """
        return self.data.head(n)

    def display_info(self):
        """
        Display the information of the DataFrame.

        Returns:
            None
        """
        return self.data.info()

    def display_description(self):
        """
        Display the statistical description of the DataFrame (transposed).

        Returns:
            pd.DataFrame: The transposed statistical description of the DataFrame.
        """
        return self.data.describe().T

    def check_null_values(self):
        """
        Check for null values in the DataFrame.

        Returns:
            pd.Series: A series with the count of null values for each column.
        """
        return self.data.isnull().sum()

    def check_duplicates(self):
        """
        Check for duplicate rows in the DataFrame.

        Returns:
            int: The number of duplicate rows in the DataFrame.
        """
        return self.data.duplicated().sum()

    def check_outliers(self):
        """
        Identify potential outliers in all numerical columns using the IQR method.

        Returns:
            pd.DataFrame: Rows containing potential outliers for all numerical columns.
        """
        outliers = pd.DataFrame()
        for column in self.data.select_dtypes(include=['number']).columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            outliers = pd.concat([outliers, column_outliers], axis=0)
        return outliers.drop_duplicates()