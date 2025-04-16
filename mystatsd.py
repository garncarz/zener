from statsd import StatsClient

from settings import settings

if settings.statsd_host:
    statsd = StatsClient(
        host=settings.statsd_host,
        port=settings.statsd_port,
        prefix=settings.app_name,
    )
else:
    from unittest.mock import MagicMock

    statsd = MagicMock()
