from google.cloud import aiplatform
import datetime
import logging
import yaml

def submit_training_job():
    """Submit training job to Vertex AI."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize Vertex AI
        aiplatform.init(
            project='sheetz-poc',
            location='us-central1',
            staging_bucket='gs://sheetz-training-artifacts'
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f'sheetz_recommender_{timestamp}'
        
        logger.info(f"Creating CustomJob: {job_name}")

        # Worker pool specification with simplified configuration
        worker_pool_specs = [{
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/sheetz-poc/sheetz-training/recommender-training:latest",
                "command": ["python", "-m", "src.train"],
                "env": [
                    {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "max_split_size_mb:512"},
                    {"name": "NCCL_DEBUG", "value": "INFO"},
                    {"name": "GOOGLE_CLOUD_PROJECT", "value": "sheetz-poc"} # Add project ID
                ]
            },
            "python_package_spec": None
        }]

        # Add service account configuration
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=f'gs://sheetz-training-artifacts/jobs/{job_name}',
        )

        logger.info("Starting job execution...")

        # Run job with simpler configuration
        job.run(
            sync=True,
            service_account=f"vertex-training@sheetz-poc.iam.gserviceaccount.com"  # Add service account
        )

        logger.info(f"Job completed with state: {job.state}")
        return job

    except Exception as e:
        logger.error(f"Error in job submission: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        submit_training_job()
    except Exception as e:
        print(f"Failed to submit job: {str(e)}")