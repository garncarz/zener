from datadog import initialize, statsd

from settings import settings

if settings.statsd_host:
    initialize(
        statsd_host=settings.statsd_host,
        statsd_port=settings.statsd_port,
        statsd_namespace=settings.app_name,
    )
else:
    from unittest.mock import MagicMock

    statsd = MagicMock()
