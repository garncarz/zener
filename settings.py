import logging
import os

from pydantic import Field, FilePath
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    input_signals: FilePath = 'Rsig-Samp.txt'

    noise_range: float = 0.005

    num_generations: int = 3

    parallel_processing: int = Field(default_factory=lambda: len(os.sched_getaffinity(0)))

    ga_filename: str = 'ga_instance'


logging.basicConfig(level=logging.INFO)
settings = Settings()
