from ucm.integration.vllm.ucm_connector import UCMDirectConnector


class _Closable:
    def __init__(self):
        self.calls = 0

    def close(self):
        self.calls += 1


def test_latest_ucm_direct_connector_shutdown_closes_unique_stores():
    connector = object.__new__(UCMDirectConnector)
    store = _Closable()
    connector.store = store
    connector.rope_store = store

    connector.shutdown()

    assert store.calls == 1
