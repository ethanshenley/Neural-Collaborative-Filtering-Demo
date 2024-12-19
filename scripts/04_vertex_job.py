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
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "gcr.io/sheetz-poc/recommender-training:latest",
                "command": ["python", "-m", "src.train"],
                "env": [
                    {"name": "GOOGLE_CLOUD_PROJECT", "value": "sheetz-poc"}
                ]
            }
        }]

        # Create custom job
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=f'gs://sheetz-training-artifacts/jobs/{job_name}'
        )

        logger.info("Starting job execution...")

        # Run job with simpler configuration
        job.run(
            sync=True,
            service_account="compute-engine-sa@sheetz-poc.iam.gserviceaccount.com"
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