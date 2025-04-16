import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    app_name: str = 'Zener'

    input_signals: str = 'Rsig-Samp.txt'

    noise_range: float = 0.005

    num_generations: int = 3

    parallel_processing: int = Field(default_factory=lambda: len(os.sched_getaffinity(0)))

    ga_filename: str = 'ga_instance'

    statsd_host: str
    statsd_port: int = 8125


logging.basicConfig(level=logging.INFO)
settings = Settings()
