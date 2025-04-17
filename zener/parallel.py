from concurrent.futures import ProcessPoolExecutor
import logging

from .settings import settings

logger = logging.getLogger(__name__)

if settings.parallel > 1:
    process_pool = ProcessPoolExecutor(max_workers=settings.parallel)
    logger.info(f'Using up to {settings.parallel} parallel processes.')
else:
    process_pool = None
