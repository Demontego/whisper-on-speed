from .celery_app import celery
import time

@celery.task(
    bind=True,
    time_limit=3*60*60,
    soft_time_limit=10200,  # 2ч50м
    acks_late=True,
    autoretry_for=(Exception,),
    max_retries=3
)
def process_audio(self, bucket: str, path: str):
    try:
        # Вызов Whisper
        time.sleep(10)
        
        if "fail" in path:
            raise ValueError("Simulated processing error")
            
        return {
            "status": "completed",
            "result": f"gs://{bucket}/{path}.transcript.txt"
        }
    except Exception as e:
        self.retry(exc=e, countdown=60)