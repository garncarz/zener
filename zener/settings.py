import logging
import platform
import os
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    app_name: str = 'Zener'
    host: str = Field(default_factory=lambda: platform.node())
    environment: Optional[str] = 'dev'

    job_id: Optional[str] = Field(default=None, alias='PBS_JOBID')

    input_signals: str = 'Rsig-Samp.txt'

    noise_range: float = 0.005

    num_generations: int = 3
    stop_fitness: float = 0
    stop_saturation: int = 100

    dynamic_population: bool = False
    dynamic_population_initial: int = 2000
    dynamic_population_final: int = 20

    parallel: int = Field(default_factory=lambda: len(os.sched_getaffinity(0)))
    use_celery: bool = False
    celery_broker_dir: str = Field(default_factory=lambda: os.path.join(os.getcwd(), 'celery_broker'))
    celery_results_dir: str = Field(default_factory=lambda: os.path.join(os.getcwd(), 'celery_results'))

    @validator('celery_broker_dir', 'celery_results_dir', pre=True)
    def validate_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

    ga_filename: str = 'ga_instance'

    dd_api_key: Optional[str] = None
    dd_site: Optional[str] = 'datadoghq.eu'

    sentry_dsn: Optional[str] = None

    # Model complexity setting
    n_f: int = 19  # Number of Kelvin-Voigt elements


logging.basicConfig(level=logging.INFO)

# To prevent the following spamming:
# WARNING:kombu.connection:No hostname was supplied. Reverting to default 'localhost'
logging.getLogger('kombu.connection').setLevel(logging.ERROR)

settings = Settings()

if settings.sentry_dsn:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
        )
        sentry_sdk.set_tag('host', settings.host)
    except Exception:
        logging.getLogger(__name__).exception('Unable to initialize Sentry; continuing without telemetry')
