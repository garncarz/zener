from contextlib import contextmanager
import logging
import time

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_resource import MetricResource
from datadog_api_client.v2.model.metric_series import MetricSeries

from .settings import settings

logger = logging.getLogger(__name__)


class BatchStatsD:

    def __init__(self):
        self._batch = None

        self._configuration = Configuration()
        self._configuration.api_key['apiKeyAuth'] = settings.dd_api_key
        self._configuration.server_variables['site'] = settings.dd_site

        self._api_client = ApiClient(self._configuration)
        self._api_instance = MetricsApi(self._api_client)

    def gauge(self, metric, value, tags=None, host=None, timestamp=None):
        if timestamp is None:
            timestamp = int(time.time())

        series = MetricSeries(
            metric=f'{settings.app_name}.{metric}',
            type=MetricIntakeType.GAUGE,
            points=[
                MetricPoint(
                    timestamp=timestamp,
                    value=float(value),
                ),
            ],
            resources=[
                MetricResource(
                    name=settings.host,
                    type='host',
                ),
            ],
            tags=tags or [],
        )

        if self._batch is not None:
            self._batch.append(series)
        else:
            self._send([series])

    @contextmanager
    def batch(self):
        self._batch = []
        try:
            yield
            self._flush()
        finally:
            self._batch = None

    def _flush(self):
        if self._batch:
            self._send(self._batch)

    def _send(self, series_list):
        body = MetricPayload(series=series_list)
        response = self._api_instance.submit_metrics(body=body)
        return response


if settings.dd_api_key:
    statsd = BatchStatsD()
else:
    from unittest.mock import MagicMock
    statsd = MagicMock()
    logger.info('Not using Datadog metrics')
