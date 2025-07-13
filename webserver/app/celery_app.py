from celery import Celery
from .config import settings
from google.cloud import compute_v1


celery = Celery(
    "tasks",
    broker=f"redis://{settings.redis_host}:{settings.redis_port}/0",
    backend=f"redis://{settings.redis_host}:{settings.redis_port}/1",
    include=["app.tasks"]
)

celery.autodiscover_tasks(["app.tasks"])

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
)

def check_queue_empty():
    inspector = celery.control.inspect()
    stats = inspector.stats()
    
    if not stats:
        return True
        
    active_tasks = sum(len(worker['active']) for worker in stats.values())
    reserved_tasks = sum(len(worker['reserved']) for worker in stats.values())
    
    return active_tasks == 0 and reserved_tasks == 0

def shutdown_gcp_instance():
    client = compute_v1.InstancesClient()
    request = compute_v1.StopInstanceRequest(
        project=settings.gcp_settings.project_id,
        zone=settings.gcp_settings.zone,
        instance=settings.gcp_settings.instance_name
    )
    client.stop(request=request)