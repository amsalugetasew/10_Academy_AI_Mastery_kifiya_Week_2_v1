import pandas as pd

class UserExperiance:
    def __init__(self, df):
        self.df = df
    

    def aggregate_handset_metrics(self, df):
        """
        Aggregate metrics per handset type.

        Parameters:
            df (pd.DataFrame): DataFrame containing the following columns:
                - 'Most Common Handset Type'
                - 'Average DL Throughput'
                - 'Average UL Throughput'
                - 'TCP DL Retrans. Vol (Bytes)'
                - 'TCP UL Retrans. Vol (Bytes)'

        Returns:
            pd.DataFrame: Aggregated DataFrame with calculated metrics per handset type.
        """
        # Aggregate the metrics per handset type
        handset_metrics = df.groupby(['MSISDN/Number','Most Common Handset Type']).agg({
            'Average DL Throughput': 'mean',
            'Average UL Throughput': 'mean',
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean'
        }).reset_index()

        # Calculate average throughput per handset type
        handset_metrics['Average Throughput'] = (
            handset_metrics['Average DL Throughput'] + handset_metrics['Average UL Throughput']
        ) / 2

        # Rename columns for clarity
        handset_metrics.rename(columns={
            'TCP DL Retrans. Vol (Bytes)': 'Avg TCP DL Retransmission',
            'TCP UL Retrans. Vol (Bytes)': 'Avg TCP UL Retransmission'
        }, inplace=True)

        return handset_metrics