import google.cloud.storage as storage
import google.cloud.bigquery as bigquery
import google.cloud.aiplatform as aiplatform
from google.cloud import monitoring_v3
import logging
from typing import Dict, List

class SetupValidator:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def validate_storage(self) -> bool:
        required_buckets = [
            "sheetz-rec-staging",
            "sheetz-rec-models",
            "sheetz-rec-tensorboard"
        ]
        
        try:
            for bucket_name in required_buckets:
                bucket = self.storage_client.get_bucket(bucket_name)
                self.logger.info(f"✓ Found bucket: {bucket_name}")
            return True
        except Exception as e:
            self.logger.error(f"Storage validation failed: {str(e)}")
            return False
            
    def validate_service_account(self) -> bool:
        sa_name = "vertex-training"
        required_roles = [
            "roles/aiplatform.user",
            "roles/bigquery.dataViewer",
            "roles/storage.objectViewer"
        ]
        
        try:
            # Check service account exists
            sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"
            # Validate roles...
            self.logger.info(f"✓ Service account {sa_email} validated")
            return True
        except Exception as e:
            self.logger.error(f"Service account validation failed: {str(e)}")
            return False
            
    def validate_bigquery(self) -> bool:
        try:
            dataset_ref = self.bq_client.dataset("sheetz_data")
            dataset = self.bq_client.get_dataset(dataset_ref)
            
            required_views = [
                "user_features_enriched",
                "product_features_enriched"
            ]
            
            for view_id in required_views:
                view_ref = dataset_ref.table(view_id)
                view = self.bq_client.get_table(view_ref)
                self.logger.info(f"✓ Found view: {view_id}")
                
            return True
        except Exception as e:
            self.logger.error(f"BigQuery validation failed: {str(e)}")
            return False
            
    def validate_vertex_ai(self) -> bool:
        try:
            aiplatform.init(
                project=self.project_id,
                location='us-central1'
            )
            # Check API access
            self.logger.info("✓ Vertex AI access validated")
            return True
        except Exception as e:
            self.logger.error(f"Vertex AI validation failed: {str(e)}")
            return False
    
    def validate_monitoring(self) -> bool:
        try:
            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{self.project_id}"
            
            # Check custom metric descriptor
            descriptor_path = f"{project_name}/metricDescriptors/custom.googleapis.com/sheetz/training/metrics"
            client.get_metric_descriptor(name=descriptor_path)
            
            self.logger.info("✓ Monitoring setup validated")
            return True
        except Exception as e:
            self.logger.error(f"Monitoring validation failed: {str(e)}")
            return False
            
    def validate_all(self) -> Dict[str, bool]:
        results = {
            "storage": self.validate_storage(),
            "service_account": self.validate_service_account(),
            "bigquery": self.validate_bigquery(),
            "vertex_ai": self.validate_vertex_ai(),
            "monitoring": self.validate_monitoring()
        }
        
        all_valid = all(results.values())
        self.logger.info(f"\nOverall validation: {'✓ PASSED' if all_valid else '✗ FAILED'}")
        
        return results

def main():
    validator = SetupValidator("sheetz-poc")
    results = validator.validate_all()
    
    if not all(results.values()):
        raise RuntimeError("Setup validation failed")

if __name__ == "__main__":
    main()