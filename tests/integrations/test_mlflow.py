import pytest

from garden_ai import GardenClient, Model
from garden_ai.mlmodel import (
    LocalModel,
    SerializationFormatException,
    stage_model_for_upload,
)
from tests.fixtures.helpers import get_fixture_file_path  # type: ignore


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
@pytest.mark.parametrize("serialize_type", [None, "pickle", "joblib", "keras"])
def test_mlflow_sklearn_register(tmp_path, toy_sklearn_model, serialize_type):
    # as if model.pkl already existed on disk
    import pickle
    import joblib  # type: ignore

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "model.pkl"
    model_path.touch()

    if serialize_type is None:
        with open(model_path, "wb") as f_out:
            pickle.dump(toy_sklearn_model, f_out)
    elif serialize_type == "pickle":
        with open(model_path, "wb") as f_out:
            pickle.dump(toy_sklearn_model, f_out)
    elif serialize_type == "joblib":
        with open(model_path, "wb") as f_out:
            joblib.dump(toy_sklearn_model, f_out)

    # simulate `$ garden-ai model register test-model-name tmp_path/model.pkl`
    name = "sk-test-model-name"

    # actually register the model
    client = GardenClient()
    local_model = LocalModel(
        local_path=str(model_path),
        model_name=name,
        flavor="sklearn",
        serialize_type=serialize_type,
        user_email="foo@example.com",
    )

    if serialize_type == "keras":
        # Assert that the 'SerializationFormatException' is raised
        with pytest.raises(SerializationFormatException):
            client.register_model(local_model)
    else:
        registered_model = client.register_model(local_model)
        # all mlflow models will have a 'predict' method
        downloaded_model = Model(registered_model.full_name)
        assert hasattr(downloaded_model, "predict")


@pytest.mark.integration
def test_mlflow_pytorch_register(tmp_path, toy_pytorch_model):
    # as if model.pkl already existed on disk
    import torch  # type: ignore

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "pytorchtest.pth"
    torch.save(toy_pytorch_model, model_path, _use_new_zipfile_serialization=False)

    # simulate `$ garden-ai model register test-model-name tmp_path/pytorchtest.pt`
    name = "pt-test-model-name"
    # actually register the model
    client = GardenClient()
    local_model = LocalModel(
        local_path=str(model_path),
        model_name=name,
        flavor="pytorch",
        user_email="foo@example.com",
    )
    registered_model = client.register_model(local_model)

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(registered_model.full_name)
    assert hasattr(downloaded_model, "predict")


@pytest.mark.integration
def test_mlflow_pytorch_extra_paths(mocker, local_model, tmp_path):
    import torch  # type: ignore

    mock_log_variant = mocker.MagicMock()
    mocker.patch("mlflow.pytorch.log_model", mock_log_variant)
    mocker.patch("garden_ai.mlmodel.MODEL_STAGING_DIR", new=tmp_path)
    model_path = get_fixture_file_path("fixture_models/pytorchtest.pth")
    file_path = get_fixture_file_path("fixture_models/torch.py")
    local_model = LocalModel(
        model_name="test_model",
        flavor="pytorch",
        local_path=str(model_path),
        user_email="willengler@uchicago.edu",
        extra_paths=[str(file_path)],
    )
    staged_path = stage_model_for_upload(local_model)
    assert staged_path.endswith("/artifacts/model")
    expected_call = mocker.call(
        torch.load(model_path),
        "model",
        registered_model_name=local_model.mlflow_name,
        code_paths=local_model.extra_paths,
        metadata={"garden_load_strategy": "pytorch"},
    )
    print(str(mock_log_variant.call_args))
    print(str(expected_call))
    assert str(mock_log_variant.call_args) == str(expected_call)


@pytest.mark.integration
@pytest.mark.parametrize("save_format", ["tf", "h5"])
def test_mlflow_tensorflow_register(tmp_path, toy_tensorflow_model, save_format):
    # as if model.pkl already existed on disk

    tmp_path.mkdir(exist_ok=True)
    model_path = tmp_path / "tensorflowtest"
    toy_tensorflow_model.save(model_path, save_format=save_format)

    # simulate `$ garden-ai model register test-model-name tmp_path/tensorflowtest`
    name = "tf-test-model-name"
    # actually register the model
    client = GardenClient()
    local_model = LocalModel(
        local_path=str(model_path),
        model_name=name,
        flavor="tensorflow",
        user_email="foo@example.com",
    )
    registered_model = client.register_model(local_model)

    # all mlflow models will have a 'predict' method
    downloaded_model = Model(registered_model.full_name)
    assert hasattr(downloaded_model, "predict")
