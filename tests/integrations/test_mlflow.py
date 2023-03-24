import pytest

from garden_ai import GardenClient, Model


@pytest.fixture(autouse=True)
def do_not_set_mlflow_env_variables():
    """overrides same fixture in ../conftest.py for this module only"""
    return


@pytest.fixture
def toy_sklearn_model():
    # import sklearn
    from sklearn import tree  # type: ignore
    from sklearn.datasets import load_wine  # type: ignore

    wine = load_wine()
    sk_model = tree.DecisionTreeClassifier().fit(wine.data, wine.target)
    return sk_model


@pytest.fixture
def toy_tensorflow_model():
    # import tensorflow
    import tensorflow as tf  # type: ignore

    tf_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=10, input_shape=[8]),
            tf.keras.layers.Dense(units=1),
        ]
    )
    tf_model.compile(loss="mse", optimizer=tf.keras.optimizers.RMSprop(0.001))
    return tf_model


@pytest.fixture
def toy_pytorch_model():
    # import torch
    import torch.nn as nn

    pytorch_model = nn.Sequential(nn.Linear(8, 10), nn.ReLU(), nn.Linear(10, 1))
    return pytorch_model


@pytest.fixture
def toy_model(request, toy_sklearn_model, toy_tensorflow_model, toy_pytorch_model):
    model_type = request.param
    if model_type == "sklearn":
        return (toy_sklearn_model, model_type)
    elif model_type == "tensorflow":
        return (toy_tensorflow_model, model_type)
    elif model_type == "pytorch":
        return (toy_pytorch_model, model_type)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "toy_model", ["sklearn", "tensorflow", "pytorch"], indirect=True
)
def test_mlflow_register(tmp_path, toy_model):
    # as if model.pkl already existed on disk
    import pickle

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "model.pkl"
    model_path.touch()
    # model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f_out:
        pickle.dump(toy_model[0], f_out)

    # simulate `$ garden-ai model register test-model-name tmp_path/model.pkl`
    name = "test-model-name"
    flavor = toy_model[1]
    extra_pip_requirements = None
    # actually register the model
    client = GardenClient()
    full_model_name = client.log_model(
        str(model_path), name, flavor, extra_pip_requirements
    )

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(full_model_name)
    assert hasattr(downloaded_model, "predict")
