import pytest

from garden_ai import GardenClient, Model


@pytest.fixture(autouse=True)
def do_not_set_mlflow_env_variables():
    """overrides same fixture in ../conftest.py for this module only"""
    return


@pytest.fixture
def toy_sklearn_model():
    # import sklearn
    from sklearn import tree
    from sklearn.datasets import load_wine

    wine = load_wine()
    sk_model = tree.DecisionTreeClassifier().fit(wine.data, wine.target)
    return sk_model


@pytest.mark.integration
def test_mlflow_register(tmp_path, toy_sklearn_model):
    # as if model.pkl already existed on disk
    import pickle

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "model.pkl"
    model_path.touch()
    # model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f_out:
        pickle.dump(toy_sklearn_model, f_out)

    # simulate `$ garden-ai model register test-model-name tmp_path/model.pkl`
    name = "test-model-name"
    extra_pip_requirements = None
    # actually register the model
    client = GardenClient()
    full_model_name = client.log_model(str(model_path), name, extra_pip_requirements)

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(full_model_name)
    assert hasattr(downloaded_model, "predict")
