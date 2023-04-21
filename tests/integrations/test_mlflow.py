import pytest

from garden_ai import GardenClient, Model


@pytest.fixture(autouse=True)
def do_not_set_mlflow_env_variables():
    """overrides same fixture in ../conftest.py for this module only"""
    return


@pytest.fixture
def toy_sklearn_model():
    from sklearn import tree  # type: ignore
    from sklearn.datasets import load_wine  # type: ignore

    wine = load_wine()
    sk_model = tree.DecisionTreeClassifier().fit(wine.data, wine.target)
    return sk_model


@pytest.fixture
def toy_pytorch_model():
    import torch  # type: ignore

    pt_model = torch.nn.Linear(6, 1)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-4)

    X = torch.randn(6)
    y = torch.randn(1)

    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = pt_model(X)

        loss = loss_function(outputs, y)
        loss.backward()

        optimizer.step()
    return pt_model


@pytest.fixture
def toy_tensorflow_model():
    import tensorflow as tf  # type: ignore
    import numpy as np  # type: ignore

    # Define the model
    tf_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, input_shape=(4,), activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    tf_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Create some dummy data for training and testing
    x_train = np.random.rand(100, 4)
    y_train = np.random.randint(2, size=(100, 1))
    x_test = np.random.rand(50, 4)
    y_test = np.random.randint(2, size=(50, 1))

    # Train the model silently
    tf_model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0,
    )
    return tf_model


@pytest.mark.integration
def test_mlflow_sklearn_register(tmp_path, toy_sklearn_model):
    # as if model.pkl already existed on disk
    import pickle

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "model.pkl"
    model_path.touch()
    flavor = "sklearn"

    with open(model_path, "wb") as f_out:
        pickle.dump(toy_sklearn_model, f_out)

    # simulate `$ garden-ai model register test-model-name tmp_path/model.pkl`
    name = "sk-test-model-name"
    extra_pip_requirements = None
    # actually register the model
    client = GardenClient()
    full_model_name = client.log_model(
        str(model_path), name, flavor, extra_pip_requirements
    )

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(full_model_name)
    assert hasattr(downloaded_model, "predict")


@pytest.mark.integration
def test_mlflow_pytorch_register(tmp_path, toy_pytorch_model):
    # as if model.pkl already existed on disk
    import torch  # type: ignore

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "pytorchtest.pth"
    torch.save(toy_pytorch_model, model_path, _use_new_zipfile_serialization=False)
    flavor = "pytorch"

    # simulate `$ garden-ai model register test-model-name tmp_path/pytorchtest.pt`
    name = "pt-test-model-name"
    extra_pip_requirements = None
    # actually register the model
    client = GardenClient()
    full_model_name = client.log_model(
        str(model_path), name, flavor, extra_pip_requirements
    )

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(full_model_name)
    assert hasattr(downloaded_model, "predict")


@pytest.mark.integration
def test_mlflow_tensorflow_register(tmp_path, toy_tensorflow_model):
    # as if model.pkl already existed on disk

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "tensorflowtest"
    toy_tensorflow_model.save(model_path)
    flavor = "tensorflow"

    # simulate `$ garden-ai model register test-model-name tmp_path/tensorflowtest`
    name = "tf-test-model-name"
    extra_pip_requirements = None
    # actually register the model
    client = GardenClient()
    full_model_name = client.log_model(
        str(model_path), name, flavor, extra_pip_requirements
    )

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(full_model_name)
    assert hasattr(downloaded_model, "predict")
