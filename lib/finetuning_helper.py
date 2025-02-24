import os
import json
from time import sleep
from openai import OpenAI
from lib.io import save_to_json

import logging
logger = logging.getLogger(__name__)

client = OpenAI()


class FineTuningHelper:
    def __init__(self, config) -> None:
        self.config = config

    def run(self, wait_for_job=False, skip_if_exist=True):
        if skip_if_exist and self.is_job_exists():
            logger.info("Model already trained, skip.")
            return
        logger.info("Uploading data...")
        file_ids = self.upload_data(
            training_file_name=self.config.dataset_train_filename,
            validation_file_name=self.config.dataset_val_filename,
        )
        logger.info("Starting training...")
        job_id = self.start_training(
            training_file_id=file_ids["training_file_id"],
            validation_file_id=file_ids["validation_file_id"],
            suffix_name=self.config.model_suffix,
        )
        if wait_for_job:
            logger.info("Waiting for training job...")
            self.wait_for_training_job(job_id=job_id)

    def upload_data(self, training_file_name, validation_file_name):
        if os.path.exists(self.config.file_id_filename):
            file_ids = json.load(open(self.config.file_id_filename, "r"))
            logger.info(f"File already uploaded, load file IDs: {file_ids}")
            return file_ids

        train_file_obj = client.files.create(
            file=open(training_file_name, "rb"), purpose="fine-tune"
        )
        training_file_id = train_file_obj.id

        validation_file_obj = client.files.create(
            file=open(validation_file_name, "rb"), purpose="fine-tune"
        )
        validation_file_id = validation_file_obj.id

        file_ids = {
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
        }
        logger.info(f"File IDs: {file_ids}")
        save_to_json(file_ids, self.config.file_id_filename)
        return file_ids

    def start_training(self, training_file_id, validation_file_id, suffix_name):
        # Create a Fine Tuning Job
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.config.fine_tuning_base_model_id,
            suffix=suffix_name,
        )

        logger.info(f"Job: {job}")
        self.save_job(job)
        return job.id

    def list_jobs(self):
        jobs = client.fine_tuning.jobs.list()
        return jobs.data

    def is_job_exists(self):
        return os.path.exists(self.config.job_id_filename)

    def is_training_succeeded(self):
        job = self.try_load_job()
        return job is not None and job.status == "succeeded"
    
    def try_load_job(self):
        if not self.is_job_exists():
            logger.warning("Model not trained yet.")
            return None
        job = json.load(open(self.config.job_id_filename, "r"))
        return job
    
    def retrieve_job(self):
        job = self.try_load_job()
        if job is None:
            return None
        job_id = job["id"]
        job = client.fine_tuning.jobs.retrieve(job_id)
        self.save_job(job)
        return job

    def wait_for_training_job(self, job_id):
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info(job)
        self.save_job(job)

        while job.status not in ("succeeded", "failed", "cancelled"):
            event_resp = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=30
            )

            events = event_resp.data
            events.reverse()

            for event in events:
                logger.info(event.message)

            logger.info("Waiting for 60 seconds...")
            sleep(60)
            job = client.fine_tuning.jobs.retrieve(job_id)
            logger.info(job)

        self.save_job(job)

        return job.status == "succeeded"

    def save_job(self, job):
        data = dict(job)
        if "hyperparameters" in data:
            del data["hyperparameters"]
        save_to_json(data, self.config.job_id_filename)
