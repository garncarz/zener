from unittest.mock import Mock

from zener.statsd import BatchStatsD


def test_datadog_failure_does_not_escape_batch(monkeypatch):
    statsd = BatchStatsD()
    statsd._api_instance = Mock()
    statsd._api_instance.submit_metrics.side_effect = ConnectionError('network unavailable')

    with statsd.batch():
        statsd.gauge('generation', 1)

    statsd._api_instance.submit_metrics.assert_called_once()


def test_datadog_failure_does_not_escape_direct_send(monkeypatch):
    statsd = BatchStatsD()
    statsd._api_instance = Mock()
    statsd._api_instance.submit_metrics.side_effect = ConnectionError('network unavailable')

    statsd.gauge('generation', 1)

    statsd._api_instance.submit_metrics.assert_called_once()