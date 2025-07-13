from .celery_app import celery  # noqa
from .config import settings  # noqa

# Реэкспорт основных компонентов для удобного импорта
__all__ = [
    'celery',
    'settings',
]

# Инициализация логгера (опционально)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)