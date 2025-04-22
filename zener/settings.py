import logging
import platform
import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    app_name: str = 'Zener'
    host: str = Field(default_factory=lambda: platform.node())

    input_signals: str = 'Rsig-Samp.txt'

    noise_range: float = 0.005

    num_generations: int = 3

    parallel: int = Field(default_factory=lambda: len(os.sched_getaffinity(0)))

    ga_filename: str = 'ga_instance'

    dd_api_key: Optional[str] = None


logging.basicConfig(level=logging.INFO)
settings = Settings()
