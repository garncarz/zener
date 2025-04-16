from statsd import StatsClient

from .settings import settings

# cannot be global because of parallelism/serialization:
def get_statsd():
    if settings.statsd_host:
        statsd = StatsClient(
            host=settings.statsd_host,
            port=settings.statsd_port,
            prefix=settings.app_name,
        )
    else:
        from unittest.mock import MagicMock

        statsd = MagicMock()

    return statsd

def gauge(name, val):
    get_statsd().gauge(name, val)
