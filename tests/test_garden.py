from garden import GardenClient

def test_client():
    gc = GardenClient(name="test", version="0.0.1")
    assert gc.name == "test"
    assert gc.version == "0.0.1"