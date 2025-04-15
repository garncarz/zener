from pydantic import FilePath
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    input_signals: FilePath = 'Rsig-Samp.txt'

    noise_range: float = 0.005

    num_generations: int = 3


settings = Settings()
