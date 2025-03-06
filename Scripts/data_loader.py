import os
import pandas as pd

class FileLoader:
    def __init__(self, file_paths):
        """Initializes the FileLoader with a list of file paths."""
        self.file_paths = file_paths
        self.dataframes = {}

    def load_files(self):
        """Loads all specified CSV and Excel files into separate dataframes."""
        if not self.file_paths:
            print("No file paths provided!")
            return

        # Loop through each file and load it into a DataFrame
        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)

            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                else:
                    print(f"Unsupported file format: {file_name}")
                    continue

                # Check if the DataFrame is empty
                if df.empty:
                    print(f"Warning: {file_name} is empty and won't be added.")
                    continue

                # Store the dataframe in a dictionary with the file name as the key
                self.dataframes[file_name] = df

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    def get_dataframe(self, file_name):
        """Retrieves a specific dataframe by file name."""
        return self.dataframes.get(file_name, None)

    def load_single_file(self, file_path):
        """Loads a single CSV or Excel file from a given path and converts it to a DataFrame."""
        try:
            # Check if the file exists and is a valid type
            if not os.path.isfile(file_path) or not file_path.endswith(('.csv', '.xls', '.xlsx')):
                raise ValueError("The provided file path is invalid or not a supported file type.")

            # Load the file into a DataFrame
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Check if the DataFrame is empty
            if df.empty:
                print(f"Warning: The file at {file_path} is empty.")
                return None

            return df

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
