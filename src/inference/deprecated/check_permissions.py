# check_permissions.py
from google.cloud import bigquery

def check_permissions():
    # Get BQ client
    client = bigquery.Client()
    
    # Print project info
    print(f"Project: {client.project}")
    print(f"Location: {client.location}")
    
    # List datasets you have access to
    try:
        datasets = list(client.list_datasets())
        print("\nDatasets:")
        for dataset in datasets:
            print(f"- {dataset.dataset_id}")
    except Exception as e:
        print(f"Error listing datasets: {str(e)}")

if __name__ == "__main__":
    check_permissions()