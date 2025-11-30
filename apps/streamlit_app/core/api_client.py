from __future__ import annotations

import requests


class TrainAPI:
    """Small helper to wrap training API calls."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def start_job(self, payload):
        resp = requests.post(f"{self.base}/train", json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def status(self, job_id: str):
        resp = requests.get(f"{self.base}/train/status", params={"id": job_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def inference_status(self, job_id: str):
        resp = requests.get(f"{self.base}/inference/status", params={"id": job_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def start_inference(self, payload):
        resp = requests.post(f"{self.base}/inference", json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def start_inference_lebanon(self, payload):
        resp = requests.post(f"{self.base}/inference-lebanon", json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def fetch_result(self, project: str | None, run: str | None, job_id: str | None = None):
        params = {}
        if project:
            params["project"] = project
        if run:
            params["run"] = run
        if job_id:
            params["job_id"] = job_id
        resp = requests.get(
            f"{self.base}/results",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self):
        resp = requests.get(f"{self.base}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
