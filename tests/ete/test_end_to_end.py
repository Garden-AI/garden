import os
import sys
import shutil
import pickle
import jinja2
import uuid
import json
import functools
import requests
import subprocess
import base64

import typer
from typing import Optional
from typer.testing import CliRunner
from pathlib import Path
from datetime import datetime, timezone
from rich.prompt import Prompt
from rich import print as rich_print
import unittest.mock as mocker

import garden_ai
from garden_ai.app.main import app

import globus_sdk


# Set to command name that failed. Used for sending slack error message.
# If the test fails somehow without setting failed on, send unknown action as failure point.
failed_on = "unknown action"

# Set to true if pre build container is on. Used for sending slack error message.
fast_run = False

# Container IDs for --pre-build-container
sklearn_container_uuid = "b9cf409f-d5f2-4956-b198-0d90ffa133e6"
tf_container_uuid = "7ac4ecf4-8af6-4477-9aad-089f0a588b04"
torch_container_uuid = "44563f7c-5045-4b6b-8e3a-34f5c7fa781e"

key_store_path = Path(os.path.expanduser("~/.garden"))

garden_title = f"ETE-Test-Garden-{str(uuid.uuid4())}"

scaffolded_pipeline_folder_name = "ete_test_pipeline_title"
pipeline_template_name = "ete_pipeline_cc"

sklearn_pipeline_path = os.path.join(key_store_path, "sklearn_pipeline.py")
sklearn_pipeline_name = "ETESklearnPipeline"
sklearn_model_name = "ETE-Test-Model-Sklearn"

tf_pipeline_path = os.path.join(key_store_path, "tf_pipeline.py")
tf_pipeline_name = "ETETfPipeline"
tf_model_name = "ETE-Test-Model-Tf"

torch_pipeline_path = os.path.join(key_store_path, "torch_pipeline.py")
torch_pipeline_name = "ETETorchPipeline"
torch_model_name = "ETE-Test-Model-Torch"

sklearn_model_location = os.path.abspath("./models/sklearn_model.pkl")
tf_model_location = os.path.abspath("./models/keras_model")
torch_model_location = os.path.abspath("./models/torch_model.pth")

sklearn_input_data_location = os.path.abspath("./models/sklearn_test_input.pkl")
tf_input_data_location = os.path.abspath("./models/keras_test_input.pkl")
torch_input_data_location = os.path.abspath("./models/torch_test_input.pkl")

sklearn_model_reqs_location = os.path.abspath("./models/sklearn_requirements.txt")
tf_model_reqs_location = os.path.abspath("./models/keras_requirements.txt")
torch_model_reqs_location = os.path.abspath("./models/torch_requirements.txt")

pipeline_template_location = os.path.abspath("./templates")

example_garden_data = {
    "authors": ["Test Garden Author"],
    "title": "ETE Test Garden Title",
    "contributors": ["Test Garden Contributor"],
    "year": "2023",
    "description": "ETE Test Garden Description",
}
example_pipeline_data = {
    "authors": ["Test Pipeline Author"],
    "title": "ETE Test Pipeline Title",
    "contributors": ["Test Pipeline Contributor"],
    "year": "2023",
    "description": "ETE Test Pipeline Description",
}

local_files_list = [
    sklearn_pipeline_path,
    tf_pipeline_path,
    torch_pipeline_path,
    os.path.join(key_store_path, scaffolded_pipeline_folder_name),
    os.path.join(key_store_path, "data.json"),
    os.path.join(key_store_path, "tokens.json"),
    os.path.join(key_store_path, "model.zip"),
]

t_app = typer.Typer()


@t_app.command()
def run_garden_end_to_end(
    garden_grant: Optional[str] = typer.Option(
        default="cc",
        help="The grant type to initialize a GardenClient with. Can be cc or at.",
    ),
    model_type: Optional[str] = typer.Option(
        default="sklearn",
        help="The model types to test. Can be sklearn, tf, torch or all.",
    ),
    globus_compute_endpoint: Optional[str] = typer.Option(
        default="none",
        help="The globus compute endpoint to remote run the test pipelines on. If none, then will not test remote runs.",
    ),
    live_print_stdout: Optional[bool] = typer.Option(
        default=False,
        help="If true, will print the outputs of the test cmds to stdout.",
    ),
    pre_build_container: Optional[str] = typer.Option(
        default="none",
        help="If test should use a pre build container for a fast run. Can be sklearn, tf or torch. If none, then will build containers normally.",
    ),
    prompt_for_git_secret: Optional[bool] = typer.Option(
        default=True,
        help="If test should as needed prompt for garden client credentials or read them from user included file ./templates/git_secrets.json. If false, user MUST provide values for GARDEN_API_CLIENT_SECRET GARDEN_API_CLIENT_ID in git_secrets.json",
    ),
):
    # Set up
    rich_print("\n[bold blue]Setup ETE Test[/bold blue]\n")

    rich_print(f"Garden grant type set to: [blue]{garden_grant}[/blue]")

    run_sklearn = False
    run_tf = False
    run_torch = False
    if model_type == "all":
        run_sklearn = True
        run_tf = True
        run_torch = True
    elif model_type == "sklearn":
        run_sklearn = True
    elif model_type == "tf":
        run_tf = True
    elif model_type == "torch":
        run_torch = True

    rich_print(f"Testing with [blue]sklearn[/blue] model: {run_sklearn}")
    rich_print(f"Testing with [blue]tensorflow[/blue] model: {run_tf}")
    rich_print(f"Testing with [blue]pytorch[/blue] model: {run_torch}")

    is_gha = os.getenv("GITHUB_ACTIONS")

    runner = None
    rich_print(f"CliRunner live print set to: {live_print_stdout}")
    if live_print_stdout:
        runner = _make_live_print_runner()
    else:
        runner = CliRunner()

    rich_print(f"Pre build container set to: [blue]{pre_build_container}[/blue]")
    if pre_build_container != "none":
        # Set to true if pre build container is on. Used for sending slack errors.
        global fast_run
        fast_run = True

    # Cleanup any left over files generated from the test
    _cleanup_local_files(local_files_list)

    rich_print(f"garden_ai module location: {garden_ai.__file__}")

    rich_print("\n[bold blue]Starting ETE Test[/bold blue]\n")

    # Change working dir to .garden
    old_cwd = os.getcwd()
    os.chdir(key_store_path)

    client = None
    if garden_grant == "cc":
        # Create GardenClient with ClientCredentialsAuthorizer and patch all instances of GardenClients
        if is_gha:
            GARDEN_API_CLIENT_ID = os.getenv("GARDEN_API_CLIENT_ID", "none")
            GARDEN_API_CLIENT_SECRET = os.getenv("GARDEN_API_CLIENT_SECRET", "none")
            assert (
                GARDEN_API_CLIENT_SECRET != "none"
                and GARDEN_API_CLIENT_SECRET != "none"
            )
        else:
            if prompt_for_git_secret:
                GARDEN_API_CLIENT_ID = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_ID here "
                ).strip()
                GARDEN_API_CLIENT_SECRET = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_SECRET here "
                ).strip()
            else:
                with open(
                    os.path.join(old_cwd, "templates/git_secrets.json")
                ) as json_file:
                    git_secrets = json.load(json_file)
                GARDEN_API_CLIENT_ID = git_secrets["GARDEN_API_CLIENT_ID"]
                GARDEN_API_CLIENT_SECRET = git_secrets["GARDEN_API_CLIENT_SECRET"]

        os.environ["FUNCX_SDK_CLIENT_ID"] = GARDEN_API_CLIENT_ID
        os.environ["FUNCX_SDK_CLIENT_SECRET"] = GARDEN_API_CLIENT_SECRET

        client = _make_garden_client_with_cc(
            GARDEN_API_CLIENT_ID, GARDEN_API_CLIENT_SECRET
        )

    elif garden_grant == "at":
        # Create GardenClient normally with access token grant
        GARDEN_API_CLIENT_ID = "none"
        GARDEN_API_CLIENT_SECRET = "none"
        global pipeline_template_name
        pipeline_template_name = "ete_pipeline_at"
        client = _make_garden_client_with_at()
    else:
        raise Exception(
            "Invalid garden grant type; must be either cc (Client credential grant) or at (Access token grant)."
        )

    # Patch all instances of GardenClient with our new grant type one and run tests with patches.
    # If pre build container is true then also patch build_container method.
    if pre_build_container != "none":
        with mocker.patch(
            "garden_ai.app.garden.GardenClient"
        ) as mock_garden_gc, mocker.patch(
            "garden_ai.app.model.GardenClient"
        ) as mock_model_gc, mocker.patch(
            "garden_ai.app.pipeline.GardenClient"
        ) as mock_pipeline_gc, mocker.patch.object(
            client, "build_container"
        ) as mock_container_build:
            mock_garden_gc.return_value = client
            mock_model_gc.return_value = client
            mock_pipeline_gc.return_value = client

            if pre_build_container == "sklearn":
                mock_container_build.return_value = sklearn_container_uuid
            elif pre_build_container == "tf":
                mock_container_build.return_value = tf_container_uuid
            elif pre_build_container == "torch":
                mock_container_build.return_value = torch_container_uuid
            else:
                raise Exception(
                    "Invalid pre build container type; must be either sklearn, tf or torch."
                )

            _run_test_cmds(
                client,
                runner,
                globus_compute_endpoint,
                run_sklearn,
                run_tf,
                run_torch,
                GARDEN_API_CLIENT_ID,
                GARDEN_API_CLIENT_SECRET,
            )
    else:
        with mocker.patch(
            "garden_ai.app.garden.GardenClient"
        ) as mock_garden_gc, mocker.patch(
            "garden_ai.app.model.GardenClient"
        ) as mock_model_gc, mocker.patch(
            "garden_ai.app.pipeline.GardenClient"
        ) as mock_pipeline_gc:
            mock_garden_gc.return_value = client
            mock_model_gc.return_value = client
            mock_pipeline_gc.return_value = client

            _run_test_cmds(
                client,
                runner,
                globus_compute_endpoint,
                run_sklearn,
                run_tf,
                run_torch,
                GARDEN_API_CLIENT_ID,
                GARDEN_API_CLIENT_SECRET,
            )

    rich_print("\n[bold blue]Finished ETE Test successfully; cleaning up[/bold blue]\n")

    os.chdir(old_cwd)

    # Cleanup local files
    _cleanup_local_files(local_files_list)

    # Send run info to slack. No error in this case.
    _make_slack_message(None)


@t_app.command()
def collect_and_send_logs(
    run_type: str = typer.Option(
        default="full",
        help="skinny or full",
    ),
):
    is_gha = os.getenv("GITHUB_ACTIONS")
    if not is_gha:
        raise Exception("For github actions use only.")

    should_send = True

    git_repo = os.getenv("GITHUB_REPOSITORY")
    git_run_id = os.getenv("GITHUB_RUN_ID")
    git_run_url = f"https://github.com/{git_repo}/actions/runs/{git_run_id}/"
    git_api_url = (
        f"https://api.github.com/repos/{git_repo}/actions/runs/{git_run_id}/jobs"
    )
    git_job_data = requests.get(git_api_url).json()

    build_jobs = []
    for job in git_job_data["jobs"]:
        if "build" in job["name"]:
            build_jobs.append(job["name"])

    msg = f"*Finished*: {git_run_url}\n"

    ete_out = os.getenv("ETE_OUT")

    if ete_out is None:
        raise Exception("Failed to find output env var.")

    if ete_out != "START_BUILD":
        old_msg_base64_bytes = ete_out.encode("ascii")
        old_mgs_string_bytes = base64.b64decode(old_msg_base64_bytes)
        old_msg_string = old_mgs_string_bytes.decode("ascii")
        print(old_msg_string)
        msg_dict = json.loads(old_msg_string)
    else:
        msg_dict = {}

    total_added_msgs = 0

    for job_name, msg_string in msg_dict.items():
        build_jobs.remove(job_name)
        if msg_string == "SKINNY_JOB_SUCCESS":
            pass
        else:
            msg += msg_string
            msg += "\n \n"
            total_added_msgs += 1

    for missing_job in build_jobs:
        timeout_msg = f"*FAILURE*, end to end run: `{run_type} {missing_job}` has no stored output, most likely timed out.\n\n"
        msg += timeout_msg
        total_added_msgs += 1

    if total_added_msgs > 0:
        _send_slack_message(msg)
    else:
        rich_print(msg)


def _run_test_cmds(
    client,
    runner,
    globus_compute_endpoint,
    run_sklearn,
    run_tf,
    run_torch,
    GARDEN_API_CLIENT_ID,
    GARDEN_API_CLIENT_SECRET,
):
    # Garden create
    new_garden = _test_garden_create(example_garden_data, garden_title, runner)

    # Pipeline create
    _test_pipeline_create(
        example_pipeline_data,
        key_store_path,
        scaffolded_pipeline_folder_name,
        runner,
    )

    if run_sklearn:
        # Model register sklearn
        sklearn_model_full_name = _test_model_register(
            sklearn_model_location, "sklearn", sklearn_model_name, runner
        )
        # Pipeline make sklearn
        sklearn_pipeline_local = _make_pipeline_file(
            sklearn_pipeline_name,
            sklearn_model_full_name,
            sklearn_model_reqs_location,
            sklearn_pipeline_path,
            pipeline_template_name,
            pipeline_template_location,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register sklearn
        sklearn_pipeline = _test_pipeline_register(
            sklearn_pipeline_path,
            sklearn_pipeline_local,
            sklearn_model_full_name,
            "sklearn",
            runner,
        )
        # Add sklearn pipeline to garden
        _test_garden_add_pipeline(new_garden, sklearn_pipeline, runner)

    if run_tf:
        # Model register tensorflow
        tf_model_full_name = _test_model_register(
            tf_model_location, "tensorflow", tf_model_name, runner
        )
        # Pipeline make tensorflow
        tf_pipeline_local = _make_pipeline_file(
            tf_pipeline_name,
            tf_model_full_name,
            tf_model_reqs_location,
            tf_pipeline_path,
            pipeline_template_name,
            pipeline_template_location,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register tensorflow
        tf_pipeline = _test_pipeline_register(
            tf_pipeline_path,
            tf_pipeline_local,
            tf_model_full_name,
            "tensorflow",
            runner,
        )
        # Add tensorflow pipeline to garden
        _test_garden_add_pipeline(new_garden, tf_pipeline, runner)

    if run_torch:
        # Model register pytorch
        torch_model_full_name = _test_model_register(
            torch_model_location, "pytorch", torch_model_name, runner
        )
        # Pipeline make pytorch
        torch_pipeline_local = _make_pipeline_file(
            torch_pipeline_name,
            torch_model_full_name,
            torch_model_reqs_location,
            torch_pipeline_path,
            pipeline_template_name,
            pipeline_template_location,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register pytorch
        torch_pipeline = _test_pipeline_register(
            torch_pipeline_path,
            torch_pipeline_local,
            torch_model_full_name,
            "pytorch",
            runner,
        )
        # Add pytorch pipeline to garden
        _test_garden_add_pipeline(new_garden, torch_pipeline, runner)

    # Publish the garden
    published_garden = _test_garden_publish(new_garden, runner)

    # Search for our garden
    _test_garden_search(published_garden, runner)

    # Test run all selected pipelines on globus compute endpoint
    if globus_compute_endpoint != "none":
        if run_sklearn:
            _test_run_garden_on_endpoint(
                published_garden,
                key_store_path,
                sklearn_pipeline_name,
                sklearn_input_data_location,
                globus_compute_endpoint,
                client,
            )
        if run_tf:
            _test_run_garden_on_endpoint(
                published_garden,
                key_store_path,
                tf_pipeline_name,
                tf_input_data_location,
                globus_compute_endpoint,
                client,
            )
        if run_torch:
            _test_run_garden_on_endpoint(
                published_garden,
                key_store_path,
                torch_pipeline_name,
                torch_input_data_location,
                globus_compute_endpoint,
                client,
            )
    else:
        rich_print("Skipping remote execution on endpoint; no endpoint given.")


def _make_garden_client_with_cc(CLIENT_ID, CLIENT_SECRET):
    try:
        rich_print(
            f"{_get_timestamp()} Starting initialize GardenClient with [blue]CC[/blue] grant."
        )
        confidential_client = globus_sdk.ConfidentialAppAuthClient(
            CLIENT_ID, CLIENT_SECRET
        )
        client = garden_ai.GardenClient(auth_client=confidential_client)
        rich_print(
            f"{_get_timestamp()} Finished initializing GardenClient with [blue]CC[/blue] grant."
        )
        return client
    except Exception as error:
        global failed_on
        failed_on = "make GardenClient with CC grant"
        rich_print(
            f"{_get_timestamp()} Failed to initialize GardenClient with [blue]CC[/blue] grant."
        )
        raise error


def _make_garden_client_with_at():
    try:
        rich_print(
            f"{_get_timestamp()} Starting initialize GardenClient with [blue]AT[/blue] grant."
        )
        client = garden_ai.GardenClient()
        rich_print(
            f"{_get_timestamp()} Finished initializing GardenClient with [blue]AT[/blue] grant."
        )
        return client
    except Exception as error:
        global failed_on
        failed_on = "make GardenClient with AT grant"
        rich_print(
            f"{_get_timestamp()} Failed to initialize GardenClient with [blue]AT[/blue] grant."
        )
        raise error


def _test_garden_create(example_garden_data, unique_title, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden create[/italic red]"
        )

        raise Exception("test error")

        gardens_before = garden_ai.local_data.get_all_local_gardens()
        assert gardens_before is None

        command = [
            "garden",
            "create",
            "--title",
            unique_title,
            "--description",
            example_garden_data["description"],
            "--year",
            example_garden_data["year"],
        ]
        for name in example_garden_data["authors"]:
            command += ["--author", name]
        for name in example_garden_data["contributors"]:
            command += ["--contributor", name]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        gardens_after = garden_ai.local_data.get_all_local_gardens()
        assert len(gardens_after) == 1

        new_garden = gardens_after[0]
        assert new_garden.title == unique_title
        assert new_garden.description == example_garden_data["description"]

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden create[/italic red] with no errors."
        )
        return new_garden
    except Exception as error:
        global failed_on
        failed_on = "garden create"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]garden create[/italic red]"
        )
        raise error


def _test_garden_add_pipeline(original_garden, pipeline, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden add-pipeline[/italic red] using pipeline: [blue]{pipeline.title}[/blue]"
        )

        command = [
            "garden",
            "add-pipeline",
            "--garden",
            original_garden.doi,
            "--pipeline",
            pipeline.doi,
        ]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        garden_after_addition = garden_ai.local_data.get_local_garden_by_doi(
            original_garden.doi
        )
        local_pipelines = garden_ai.local_data.get_all_local_pipelines()
        local_pipeline_ids = []
        for pl in local_pipelines:
            local_pipeline_ids.append(pl.doi)

        for pl_id in garden_after_addition.pipeline_ids:
            assert pl_id not in original_garden.pipeline_ids
            assert pl_id in garden_after_addition.pipeline_ids
            assert pl_id in local_pipeline_ids

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden add-pipeline[/italic red] using pipeline: [blue]{pipeline.title}[/blue] with no errors"
        )
    except Exception as error:
        global failed_on
        failed_on = "garden add-pipeline"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]garden add-pipeline[/italic red] using pipeline: [blue]{pipeline.title}[/blue]"
        )
        raise error


def _test_garden_publish(garden, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden publish[/italic red]"
        )

        command = [
            "garden",
            "publish",
            "-g",
            garden.doi,
        ]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden publish[/italic red] with no errors"
        )

        return garden_ai.local_data.get_local_garden_by_doi(garden.doi)
    except Exception as error:
        global failed_on
        failed_on = "garden publish"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]garden publish[/italic red]"
        )
        raise error


def _test_garden_search(garden, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden search[/italic red]"
        )

        command = [
            "garden",
            "search",
            "-t",
            garden.title,
        ]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        assert garden.title in result.stdout
        assert str(garden.doi) in result.stdout

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden search[/italic red] with no errors"
        )
    except Exception as error:
        global failed_on
        failed_on = "garden search"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]garden search[/italic red]"
        )
        raise error


def _test_model_register(model_location, flavor, short_name, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]model register[/italic red] using model flavor: [blue]{flavor}[/blue]"
        )

        command = [
            "model",
            "register",
            short_name,
            str(model_location),
            flavor,
        ]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        local_models = garden_ai.local_data.get_all_local_models()

        local_model = None
        for model in local_models:
            if model.model_name == short_name:
                local_model = model
                break

        assert local_model is not None
        assert local_model.full_name is not None
        assert local_model.model_name == short_name
        assert local_model.flavor == flavor

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]model register[/italic red] using model flavor: [blue]{flavor}[/blue] with no errors"
        )

        return local_model.full_name
    except Exception as error:
        global failed_on
        failed_on = "model register"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]model register[/italic red] using model flavor: [blue]{flavor}[/blue]"
        )
        raise error


def _test_pipeline_create(
    example_pipeline_data, location, scaffolded_pipeline_folder_name, runner
):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]pipeline create[/italic red]"
        )

        command = [
            "pipeline",
            "create",
            "--directory",
            str(location),
            "--title",
            example_pipeline_data["title"],
            "--description",
            example_pipeline_data["description"],
            "--year",
            example_pipeline_data["year"],
        ]
        for name in example_pipeline_data["authors"]:
            command += ["--author", name]
        for name in example_pipeline_data["contributors"]:
            command += ["--contributor", name]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        assert os.path.exists(os.path.join(location, scaffolded_pipeline_folder_name))
        assert os.path.isfile(
            os.path.join(location, scaffolded_pipeline_folder_name, "pipeline.py")
        )
        assert os.path.isfile(
            os.path.join(location, scaffolded_pipeline_folder_name, "requirements.txt")
        )

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]pipeline create[/italic red] with no errors"
        )
    except Exception as error:
        global failed_on
        failed_on = "pipeline create"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]pipeline create[/italic red]"
        )
        raise error


def _test_pipeline_register(pipeline_path, pipeline, model_full_name, flavor, runner):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]pipeline register[/italic red] using model flavor: [blue]{flavor}[/blue]"
        )

        command = [
            "pipeline",
            "register",
            pipeline_path,
        ]

        result = runner.invoke(app, command)
        try:
            assert result.exit_code == 0
        except AssertionError:
            raise result.exception

        local_pipelines = garden_ai.local_data.get_all_local_pipelines()

        registered_pipeline = None
        for local_pipeline in local_pipelines:
            if str(pipeline.doi) == str(local_pipeline.doi):
                registered_pipeline = local_pipeline
                break

        assert registered_pipeline is not None
        assert pipeline.title == registered_pipeline.title
        assert registered_pipeline.doi is not None

        assert len(registered_pipeline.steps) == 1
        assert model_full_name in registered_pipeline.steps[0]["model_full_names"]

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]pipeline register[/italic red] using model flavor: [blue]{flavor}[/blue] with no errors"
        )

        return registered_pipeline
    except Exception as error:
        global failed_on
        failed_on = "pipeline register"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]pipeline register[/italic red] using model flavor: [blue]{flavor}[/blue]"
        )
        raise error


def _test_run_garden_on_endpoint(
    garden, key_store_path, pipeline_name, input_file, globus_compute_endpoint, client
):
    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}[/blue]"
        )

        with open(input_file, "rb") as f:
            Xtest = pickle.load(f)
        test_garden = client.get_published_garden(garden.doi)
        run_pipeline = getattr(test_garden, pipeline_name)

        result = run_pipeline(Xtest, endpoint=globus_compute_endpoint)

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}[/blue] with no errors"
        )
        assert result is not None
    except Exception as error:
        global failed_on
        failed_on = "run garden on remote endpoint"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]run garden remote[/italic red] using pipeline: [blue]{pipeline_name}[/blue]"
        )
        raise error


def _make_pipeline_file(
    short_name,
    model_full_name,
    req_file_path,
    save_path,
    template_name,
    pipeline_template_location,
    GARDEN_API_CLIENT_ID,
    GARDEN_API_CLIENT_SECRET,
    client,
):
    try:
        rich_print(
            f"\n{_get_timestamp()} Making pipeline file: [blue]{short_name}[/blue]"
        )

        @garden_ai.step
        def run_inference(arg: object) -> object:
            """placeholder"""
            return arg

        pipeline = client.create_pipeline(
            title=short_name,
            authors=["ETE Test Author"],
            contributors=[],
            steps=[run_inference],  # type: ignore
            tags=[],
            description="ETE Test Pipeline",
            year=str(datetime.now().year),
        )

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(pipeline_template_location)
        )
        template = env.get_template(template_name)
        contents = template.render(
            short_name=short_name,
            pipeline=pipeline,
            model_full_name=model_full_name,
            req_file_path=req_file_path,
            GARDEN_API_CLIENT_ID=GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET=GARDEN_API_CLIENT_SECRET,
        )

        with open(save_path, "w") as f:
            f.write(contents)

        rich_print(
            f"{_get_timestamp()} Finished pipeline file: [blue]{short_name}[/blue]"
        )
        return pipeline
    except Exception as error:
        global failed_on
        failed_on = "make pipeline file"
        rich_print(
            f"{_get_timestamp()} Failed to make pipeline file: [blue]{short_name}[/blue]"
        )
        raise error


def _cleanup_local_files(file_lists):
    rich_print("\nDeleting leftover up local files.")
    for path in file_lists:
        if os.path.isfile(path):
            rich_print(f"Deleting file: {path}")
            os.remove(path)
        elif os.path.isdir(path):
            rich_print(f"Deleting folder: {path}")
            shutil.rmtree(path)
        else:
            rich_print(f"Could not find path: {path}, skipping")


def _make_live_print_runner():
    """
    Yield a typer.testing.CliRunner to invoke the CLI
    This enables pytest to show the output in its logs on test failures.
    https://github.com/pallets/click/issues/737
    """

    cls = CliRunner

    def invoke_wrapper(f):
        """Augment CliRunner.invoke to emit its output to stdout."""

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            sys.stdout.write(result.output)
            return result

        return wrapper

    cls.invoke = invoke_wrapper(cls.invoke)
    cli_runner = cls()

    return cli_runner


def _make_slack_message(error):
    is_gha = os.getenv("GITHUB_ACTIONS")

    if is_gha:
        MAX_ERROR_LENGTH = 500

        git_repo = os.getenv("GITHUB_REPOSITORY")
        git_run_id = os.getenv("GITHUB_RUN_ID")
        git_job_name_ext = os.getenv("GITHUB_JOB_NAME_EXT")
        git_job_name_int = os.getenv("GITHUB_JOB_NAME_INT")

        git_api_url = (
            f"https://api.github.com/repos/{git_repo}/actions/runs/{git_run_id}/jobs"
        )
        git_job_data = requests.get(git_api_url).json()

        current_job = None
        for job in git_job_data["jobs"]:
            if job["name"] in git_job_name_int:
                current_job = job
                break
        assert current_job is not None

        start_time = datetime.strptime(
            str(current_job["started_at"]), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        start_time_str = str(start_time)
        total_time = str((datetime.now(timezone.utc) - start_time))

        if error is None:
            if not fast_run:
                msg = (
                    f"*SUCCESS*, end to end run: `{git_job_name_ext}` passed all tests."
                    f"\nStart time: `{start_time_str}` UTC, total run time: `{total_time}`"
                )
                # _send_slack_message(msg)
                _add_msg_to_outputs(msg)
            else:
                rich_print(
                    f"SUCCESS, end to end run: {git_job_name_ext} passed all tests."
                    f"\nStart time: {start_time_str} UTC, total run time: {total_time}"
                    "\nSkipping slack message for skinny run with no errors."
                )
                _add_msg_to_outputs("SKINNY_JOB_SUCCESS")
        else:
            error_body = str(error).encode("ascii", "ignore").decode("ascii")
            if len(error_body) > MAX_ERROR_LENGTH:
                error_body = f"{error_body[0:MAX_ERROR_LENGTH]}..."
            error_msg = f"{type(error).__name__}: {error_body}"
            msg = (
                f"*FAILURE*, end to end run: `{git_job_name_ext}` failed during: `{failed_on}` ```{error_msg}``` "
                f"Start time: `{start_time_str}` UTC, total run time: `{total_time}`"
            )
            # _send_slack_message(msg)
            _add_msg_to_outputs(msg)
    else:
        rich_print("Skipping slack message; not github actions run.")


def _add_msg_to_outputs(msg):
    is_gha = os.getenv("GITHUB_ACTIONS")

    if is_gha:
        ete_in_msg = os.getenv("ETE_OUT", "START_BUILD")

        rich_print(f"ETE_OUT: \n{ete_in_msg}")

        msg_dict = {}
        if ete_in_msg != "START_BUILD":
            old_msg_base64_bytes = ete_in_msg.encode("ascii")
            old_mgs_string_bytes = base64.b64decode(old_msg_base64_bytes)
            old_msg_string = old_mgs_string_bytes.decode("ascii")
            msg_dict = json.loads(old_msg_string)

        msg_key = os.getenv("GITHUB_JOB_NAME_INT")
        msg_dict[msg_key] = msg
        msg_dict_string = json.dumps(msg_dict)

        msg_bytes = msg_dict_string.encode("ascii")
        msg_base64_bytes = base64.b64encode(msg_bytes)
        msg_base64_string = msg_base64_bytes.decode("ascii")

        process = subprocess.Popen(
            f'echo "ETE_OUT={msg_base64_string}" >> "$GITHUB_ENV"',
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
        )
        process.wait()
        rich_print(f"Added to ETE_OUT base64 encoded message:\n{msg}")


def _send_slack_message(msg):
    rich_print(f"Sending msg to slack:\n{msg}")
    slack_hook = os.getenv("SLACK_HOOK_URL", "none")
    payload = '{"text": "%s"}' % msg
    requests.post(slack_hook, data=payload)


def _get_timestamp():
    current_time = str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    return f"[bold purple][{current_time}][/bold purple]"


if __name__ == "__main__":
    try:
        t_app()
    except Exception as error:
        try:
            # Catch any exceptions thown durring the test and make error msg to slack.
            _make_slack_message(error)
        except Exception as error_msger:
            # Something weird broke, just report failure.
            rich_print("Something unknown has broken. Unable to log failure.")
            raise error_msger
        else:
            raise error
