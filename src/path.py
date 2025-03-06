class FilePathManager:
    """
    A class to manage file paths.
    """

    def __init__(self):
        """
        Initialize the FilePathManager with a list of predefined file paths.
        """
        self.file_paths = [
            "../src/Data/Copy of Week2_challenge_data_source(CSV).csv",
            # "../src/Data/Copy of Week2_challenge_data_source.xlsx",
            "../src/Data/Copy of Field Descriptions.xlsx"
        ]
        
        self.file_name = [
            "Copy of Week2_challenge_data_source(CSV).csv",
            # "../src/Data/Copy of Week2_challenge_data_source.xlsx",
            "Copy of Field Descriptions.xlsx"
        ]

    def get_file_paths(self):
        """
        Retrieve the list of file paths.

        Returns:
            list: A list of file paths.
        """
        return self.file_paths

    def get_file_name(self):
        """
        Retrieve the list of file paths.

        Returns:
            list: A list of file paths.
        """
        return self.file_name
