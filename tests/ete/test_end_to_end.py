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
from typing import Optional, Dict, List, Any
from typer.testing import CliRunner
from pathlib import Path
from datetime import datetime, timezone
from rich.prompt import Prompt
from rich import print as rich_print
import unittest.mock as mocker

import garden_ai  # type: ignore
from garden_ai.app.main import app  # type: ignore

import globus_sdk

"""
Quick architecture overview:

This script consists of two runnable commands: run-garden-end-to-end and collect-and-send-logs

run-garden-end-to-end is the actual end to end test. collect-and-send-logs is for internal github actions use only.

There are two ways of running the test: manualy or through github actions.
Manual runs are run locally, test the user's local Garden code, and do not send test output to slack.
Github actions runs are started from ete_test_skinny.yml and ete_test_full.yml. These runs send test output to slack.

See pydocs below for run_garden_end_to_end various cli args.

The github actions run uses github artifacts to store output. When running on github actions,
the ETE test makes a new env var ETE_ART_ID and sets it to a random uuid. This will be the name of the file the output is stored too.
After the test finishes running, _make_run_results_msg is called, which makes the output string for that run of the test.
_make_run_results_msg then calls _add_msg_to_artifact which base64 encodes the output msg, saves it to a new file at the location
${{ env.ETE_ART_LOC }}/${{ env.ETE_ART_ID }}.txt, and sets the env var ETE_JOB_FINISHED to TRUE.

Once the workflow has finished running the end to end test step, the value of ETE_JOB_FINISHED is checked.
If ETE_JOB_FINISHED is FALSE, the workflow creates the file ${{ env.ETE_ART_LOC }}/${{ env.ETE_ART_ID }}.txt and echos a failed message into it.

The workflow then uploads the folder ${{ env.ETE_ART_LOC }} as a github artifact of the name ${{ env.ETE_ART_NAME }}. ${{ env.ETE_ART_NAME }} is
the same for all jobs in this workflow, so all job outputs will get added to this artifact.

Once the workflow has finished all the test jobs, the workflow runs collect-and-send-logs.
collect-and-send-logs grabs the artifact ${{ env.ETE_ART_NAME }} and reads the contents of all the files in it.
At this point in the test, each job should have saved an output to a file in that artifact, so once collect-and-send-logs
has decoded all the logs, the workflow can send the output to slack and the workflow is done.
"""


# Set to true if running collect_and_send_logs.
# Used by exception handeling to know if running test or collection.
is_collecting = False

# Set to command name that failed. Used for sending slack error message.
# If the test fails somehow without setting failed on, send unknown action as failure point.
failed_on = "unknown action"

# Container IDs for --pre-build-container
sklearn_container_uuid_py38 = "b9cf409f-d5f2-4956-b198-0d90ffa133e6"
sklearn_container_uuid_py39 = "946155fa-c79e-48d1-9316-42353d4f97c3"
sklearn_container_uuid_py310 = "7705f553-cc5c-4404-8e4c-a37cf718571e"

tf_container_uuid_py38 = "e6beb8d0-fef6-470d-ae8d-74e9bcbffe10"
tf_container_uuid_py39 = "58f51be9-ef56-4064-add1-f030f59da6aa"
tf_container_uuid_py310 = "c5639554-e46d-444d-adcb-3dbc1e4d6ab8"

torch_container_uuid_py38 = "63c2caec-7eb2-4930-b778-d74e13351bf6"
torch_container_uuid_py39 = "93f0d0c2-ea65-41c1-8ee8-bb6713fbec59"
torch_container_uuid_py310 = "846c5432-eed1-41bc-b469-ef7974b6598c"

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

is_gha = os.getenv("GITHUB_ACTIONS")

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
    pre_build_container: Optional[str] = typer.Option(
        default="none",
        help="If test should use a pre build container for a fast run. Can be sklearn, tf or torch. If none, then will build containers normally.",
    ),
    globus_compute_endpoint: Optional[str] = typer.Option(
        default="none",
        help="The globus compute endpoint to remote run the test pipelines on. If none, then will not test remote runs.",
    ),
    live_print_stdout: Optional[bool] = typer.Option(
        default=False,
        help="If true, will print the outputs of the test cmds to stdout.",
    ),
    prompt_for_git_secret: Optional[bool] = typer.Option(
        default=True,
        help="If test should as needed prompt for garden client credentials or read them from user included file ./templates/git_secrets.json. "
        "If false, user MUST provide values for GARDEN_API_CLIENT_SECRET GARDEN_API_CLIENT_ID in git_secrets.json",
    ),
):
    """
    Tests garden commands: 'garden create', 'garden add-pipeline', 'garden search', 'garden publish',
    'model register', 'pipeline create', 'pipeline register' and remote execution of gardens.

    Args:
        garden_grant : Optional[str]
            The globus auth grant to initilize the GardenClient.
            Can be either 'at' (access token) to login with a naive app login or
            'cc' (client credentials) to login with a client credientials login.
        model_type : Optional[str]
            The model type with which to test Garden. Can be either 'sklearn', 'tf', 'torch' or 'all'.
        pre_build_container : Optional[str]
            If set, registers a pipeline with a pre build container of the given type. Can be
            either 'sklearn', 'tf' or 'torch'. Note only use if you are running test with one model type.
        globus_compute_endpoint : Optional[str]
            The compute endpoint on which to test remote execution. If left unset, will not test remote execution.
        live_print_stdout : Optional[bool]
            If set to true, will print all CLI outputs to stdout.
        prompt_for_git_secret : Optional[bool]
            ONLY FOR CLIENT CREDENTIAL LOCAL RUNS
            If set to true, will prompt local run user to input values for GARDEN_API_CLIENT_ID
            and GARDEN_API_CLIENT_SECRET during runtime. If set to false, expects the user to manually
            include json file ./templates/git_secrets.json containing previously mentioned values.

    Returns:
        None
    """

    # Set up ETE test
    rich_print("\n[bold blue]Setup ETE Test[/bold blue]\n")

    rich_print(f"Garden grant type set to: [blue]{garden_grant}[/blue]")

    # Set what model types are being run
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

    # Remove Optional type
    assert globus_compute_endpoint is not None

    # If run with --live-print-stdout, will print all commands output to console.
    runner = None
    rich_print(f"CliRunner live print set to: {live_print_stdout}")
    if live_print_stdout:
        runner = _make_live_print_runner()
    else:
        runner = CliRunner()

    rich_print(f"Pre build container set to: [blue]{pre_build_container}[/blue]")

    # Cleanup any left over files generated from the test
    _cleanup_local_files(local_files_list)

    rich_print(f"garden_ai module location: {garden_ai.__file__}")

    rich_print("\n[bold blue]Starting ETE Test[/bold blue]\n")

    # Change working dir to .garden
    old_cwd = os.getcwd()
    os.chdir(key_store_path)

    client = None
    if garden_grant == "cc":
        # Create GardenClient with ClientCredentialsAuthorizer
        if is_gha:
            GARDEN_API_CLIENT_ID = os.getenv("GARDEN_API_CLIENT_ID", "none")
            GARDEN_API_CLIENT_SECRET = os.getenv("GARDEN_API_CLIENT_SECRET", "none")
            assert (
                GARDEN_API_CLIENT_SECRET != "none"
                and GARDEN_API_CLIENT_SECRET != "none"
            )
        else:
            if prompt_for_git_secret:
                # If run with --prompt-for-git_secret then user must provide
                # CC login secrets during non github actions run.
                GARDEN_API_CLIENT_ID = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_ID here "
                ).strip()
                GARDEN_API_CLIENT_SECRET = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_SECRET here "
                ).strip()
            else:
                # If run with --prompt-for-git_secret then user must provide
                # CC login secrets in ./templates/git_secrets.json
                with open(
                    os.path.join(old_cwd, "templates/git_secrets.json")
                ) as json_file:
                    git_secrets = json.load(json_file)
                GARDEN_API_CLIENT_ID = git_secrets["GARDEN_API_CLIENT_ID"]
                GARDEN_API_CLIENT_SECRET = git_secrets["GARDEN_API_CLIENT_SECRET"]

        # CC login with FUNCX
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

            # pre build container is true, mocking build_container
            py_version = sys.version_info
            if pre_build_container == "sklearn":
                if py_version[0] == 3 and py_version[1] == 8:
                    mock_container_build.return_value = sklearn_container_uuid_py38
                elif py_version[0] == 3 and py_version[1] == 9:
                    mock_container_build.return_value = sklearn_container_uuid_py39
                elif py_version[0] == 3 and py_version[1] == 10:
                    mock_container_build.return_value = sklearn_container_uuid_py310
                else:
                    raise Exception("Invalid python version.")
            elif pre_build_container == "tf":
                if py_version[0] == 3 and py_version[1] == 8:
                    mock_container_build.return_value = tf_container_uuid_py38
                elif py_version[0] == 3 and py_version[1] == 9:
                    mock_container_build.return_value = tf_container_uuid_py39
                elif py_version[0] == 3 and py_version[1] == 10:
                    mock_container_build.return_value = tf_container_uuid_py310
                else:
                    raise Exception("Invalid python version.")
            elif pre_build_container == "torch":
                if py_version[0] == 3 and py_version[1] == 8:
                    mock_container_build.return_value = torch_container_uuid_py38
                elif py_version[0] == 3 and py_version[1] == 9:
                    mock_container_build.return_value = torch_container_uuid_py39
                elif py_version[0] == 3 and py_version[1] == 10:
                    mock_container_build.return_value = torch_container_uuid_py310
                else:
                    raise Exception("Invalid python version.")
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
    _make_run_results_msg(None)


@t_app.command()
def collect_and_send_logs():
    """
    Reads contents of all files in artifact directory and assembles/sends to slack final run output.

    Args:
        None

    Returns:
        None
    """

    # Sets global is_collecting to true so exception handeling knows we are running
    # collect_and_send_logs and not run_garden_end_to_end
    global is_collecting
    is_collecting = True

    if not is_gha:
        raise Exception("For github actions use only.")
    try:
        ete_out_path = os.getenv("ETE_ART_LOC")
        if ete_out_path is None:
            raise Exception("Failed to find output artifact file.")

        git_repo = os.getenv("GITHUB_REPOSITORY")
        git_run_id = os.getenv("GITHUB_RUN_ID")
        git_run_url = f"https://github.com/{git_repo}/actions/runs/{git_run_id}/"

        msg = f"*Finished*: {git_run_url}\n"

        # All jobs make an output file and adds to artifact folder.
        # Get all files in folder and add contents to msg
        out_files = os.listdir(ete_out_path)
        total_added_msgs = 0
        for file in out_files:
            path = os.path.join(ete_out_path, file)
            if not os.path.isfile(path):
                rich_print(f"Unable to read file {str(path)}, skipping.")
                continue
            with open(path, "r") as f:
                encoded_msg = f.read()
                msg_base64_bytes = encoded_msg.encode("ascii")
                mgs_string_bytes = base64.b64decode(msg_base64_bytes)
                msg_string = mgs_string_bytes.decode("ascii")
            if msg_string == "SKINNY_JOB_SUCCESS":
                rich_print("Found skinny job success, skipping.")
                continue
            else:
                msg += msg_string
                msg += "\n\n"
                total_added_msgs += 1

        # Slack responds with invalid_payload breaks when trying to deal with escaped double quote.
        # Just change to single quote for now.
        msg = msg.replace('"', "'")

        # If total_added_msgs is less than 0, all outputs where skinny success,
        # Don't need to send to slack in this case
        if total_added_msgs > 0:
            _send_slack_message(msg)
        else:
            rich_print(msg)
    except Exception as error:
        # If something failed while trying to send outputs, send plain failure message.
        _send_failure_slack_message()
        raise error


def _run_test_cmds(
    client: garden_ai.GardenClient,
    runner: CliRunner,
    globus_compute_endpoint: str,
    run_sklearn: bool,
    run_tf: bool,
    run_torch: bool,
    GARDEN_API_CLIENT_ID: str,
    GARDEN_API_CLIENT_SECRET: str,
):
    """
    Runs all garden CLI commands after env has been setup.

    Args:
        client : garden_ai.GardenClient
            The GardenClient to use.
        runner : CliRunner
            The CliRunner to run the garden commands with.
        globus_compute_endpoint : str
            The globus compute endpoint to run remote execution test on. If 'none' will not run remote execution.
        run_sklearn : bool
            Whether to run sklearn model tests.
        run_tf : bool
            Whether to run tensorflow model tests.
        run_torch : bool
            Whether to run pytorch model tests.
        GARDEN_API_CLIENT_ID : str
            If run is client credential grant, is set to secret GARDEN_API_CLIENT_ID, otherwise is 'none'
        GARDEN_API_CLIENT_SECRET : str
            If run is client credential grant, is set to secret GARDEN_API_CLIENT_SECRET, otherwise is 'none'

    Returns:
        None
    """

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
                sklearn_pipeline_name,
                sklearn_input_data_location,
                globus_compute_endpoint,
                client,
            )
        if run_tf:
            _test_run_garden_on_endpoint(
                published_garden,
                tf_pipeline_name,
                tf_input_data_location,
                globus_compute_endpoint,
                client,
            )
        if run_torch:
            _test_run_garden_on_endpoint(
                published_garden,
                torch_pipeline_name,
                torch_input_data_location,
                globus_compute_endpoint,
                client,
            )
    else:
        rich_print("Skipping remote execution on endpoint; no endpoint given.")


def _make_garden_client_with_cc(CLIENT_ID: str, CLIENT_SECRET: str):
    """
    Makes GardenClient with a client credentials login.

    Args:
        CLIENT_ID : str
            The GARDEN_API_CLIENT_ID secret
        CLIENT_SECRET : str
            The GARDEN_API_CLIENT_SECRET secret

    Returns: GardenClient
        An instance of a GardenClient initilized with a client credentials grant
    """

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
    """
    Makes GardenClient with a naive app login.

    Args:
        CLIENT_ID : str
            The GARDEN_API_CLIENT_ID secret
        CLIENT_SECRET : str
            The GARDEN_API_CLIENT_SECRET secret

    Returns: GardenClient
        An instance of a GardenClient initilized with a access token grant
    """

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


def _test_garden_create(
    example_garden_data: Dict[Any, Any], unique_title: str, runner: CliRunner
):
    """
    Runs cmd 'garden create' and checks for errors.

    Args:
        example_garden_data : Dict[Any, Any]
            Example garden data to fill out garden create with.
        unique_title : str
            Unique title for the test garden.
        runner : CliRunner
            The CliRunner to run the garden commands with.

    Returns: garden_ai.gardens.Garden
        The instance of the new Garden we created.
    """

    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden create[/italic red]"
        )

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
            raise result.exception  # type: ignore

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


def _test_garden_add_pipeline(
    original_garden: garden_ai.gardens.Garden,
    pipeline: garden_ai.pipelines.RegisteredPipeline,
    runner: CliRunner,
):
    """
    Runs cmd 'garden add-pipeline' and checks for errors.

    Args:
        original_garden : garden_ai.gardens.Garden
            Garden created in step _test_garden_create to add new pipeline too.
        pipeline : garden_ai.pipelines.RegisteredPipeline
            New registered pipeline to add to garden.
        runner : CliRunner
            The CliRunner to run the garden commands with.

    Returns:
        None
    """

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
            raise result.exception  # type: ignore

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
            f"{_get_timestamp()} Finished test: [italic red]garden add-pipeline[/italic red] using pipeline: [blue]{pipeline.title}"
            "[/blue] with no errors"
        )
    except Exception as error:
        global failed_on
        failed_on = "garden add-pipeline"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]garden add-pipeline[/italic red] using pipeline: [blue]{pipeline.title}[/blue]"
        )
        raise error


def _test_garden_publish(garden: garden_ai.gardens.Garden, runner: CliRunner):
    """
    Runs cmd 'garden publish' and checks for errors.

    Args:
        garden : garden_ai.gardens.Garden
            Garden created in step _test_garden_create to publish
        runner : CliRunner
            The CliRunner to run the garden commands with.

    Returns: garden_ai.gardens.Garden
        The new published garden.
    """

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
            raise result.exception  # type: ignore

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


def _test_garden_search(garden: garden_ai.gardens.Garden, runner: CliRunner):
    """
    Runs cmd 'garden search' and checks for errors.

    Args:
        garden : garden_ai.gardens.Garden
            The Garden published in step _test_garden_publish to search
        runner : CliRunner
            The CliRunner to run the garden commands with.

    Returns:
        None
    """

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
            raise result.exception  # type: ignore

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


def _test_model_register(
    model_location: str, flavor: str, short_name: str, runner: CliRunner
):
    """
    Runs cmd 'model register' and checks for errors.

    Args:
        model_location : str
            The location of the model file.
        flavor : str
            The flavor of the model.
        short_name : str
            The short name to register the model under.
        runner : CliRunner
            The CliRunner to run the model commands with.

    Returns: str
        The full name of the registerd model.
    """

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
            raise result.exception  # type: ignore

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
    example_pipeline_data: Dict[Any, Any],
    location: Path,
    scaffolded_pipeline_folder_name: str,
    runner: CliRunner,
):
    """
    Runs cmd 'pipeline create' and checks for errors.

    Args:
        example_pipeline_data : Dict[Any, Any]
            The pipeline data to create the new pipeline with.
        location : Path
            The location to save the new pipeline too.
        scaffolded_pipeline_folder_name : str
            The folder name to save the new pipeline too.
        runner : CliRunner
            The CliRunner to run the pipeline commands with.

    Returns:
        None
    """
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
            raise result.exception  # type: ignore

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


def _test_pipeline_register(
    pipeline_path: str,
    pipeline: garden_ai.pipelines.Pipeline,
    model_full_name: str,
    flavor: str,
    runner: CliRunner,
):
    """
    Runs cmd 'pipeline register' and checks for errors.

    Args:
        pipeline_path : str
            The pipeline data to create the new pipeline with.
        pipeline : garden_ai.pipelines.Pipeline
            The unregisted pipeline created in _make_pipeline_file to register
        model_full_name : str
            The full name of the model in this pipeline.
        flavor : str
            The flavor the model in this pipeline.
        runner : CliRunner
            The CliRunner to run the pipeline commands with.

    Returns: garden_ai.pipelines.RegisteredPipeline
        The registered pipeline.
    """

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
            raise result.exception  # type: ignore

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
    garden: garden_ai.gardens.Garden,
    pipeline_name: str,
    input_data_file: str,
    globus_compute_endpoint: str,
    client: garden_ai.GardenClient,
):
    """
    Runs a remote execution of a garden and checks for errors.

    Args:
        garden: garden_ai.gardens.Garden
            The garden to remote execute.
        pipeline_name : str
            The name of the pipeline to execute.
        input_data_file : str
            The path to the input data for this remote execution.
        globus_compute_endpoint : str
            The the globus compute endpoint to run on.
        client : garden_ai.GardenClient,
            The GardenClient to run the remote exectution with.

    Returns:
        None
    """

    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}[/blue]"
        )

        with open(input_data_file, "rb") as f:
            Xtest = pickle.load(f)
        test_garden = client.get_published_garden(garden.doi)
        run_pipeline = getattr(test_garden, pipeline_name)

        result = run_pipeline(Xtest, endpoint=globus_compute_endpoint)

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}"
            "[/blue] with no errors"
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
    short_name: str,
    model_full_name: str,
    req_file_path: str,
    save_path: str,
    template_name: str,
    pipeline_template_location: str,
    GARDEN_API_CLIENT_ID: str,
    GARDEN_API_CLIENT_SECRET: str,
    client: garden_ai.GardenClient,
):
    """
    Makes the pipeline file to reigster.

    Args:
        short_name : str
            Short name to create pipeline with.
        model_full_name : str
            Full name of the model in the pipeline.
        req_file_path : str
            Path to requirements.txt
        save_path : str
            Path to where to save new pipeline file.
        template_name : str
            Name of the jinja2 pipeline template.
        pipeline_template_location : str
            Location of the jinja2 pipeline template.
        GARDEN_API_CLIENT_ID: str
        GARDEN_API_CLIENT_SECRET: str
        client : garden_ai.GardenClient,
            The GardenClient to run the pipeline commands with.

    Returns: garden_ai.pipelines.Pipeline
        The new pipeline created.
    """

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


def _make_run_results_msg(error: Optional[Exception]):
    """
    Makes the output message from the given error.
    If error is None, will make success message.
    Once done, gives new output message to _add_msg_to_artifact
    to store in file to upload as github artifact.

    Args:
        error : Optional[Exception]
            None or exception raised during run.

    Returns:
        None
    """

    if is_gha:
        MAX_ERROR_LENGTH = 500

        git_repo = os.getenv("GITHUB_REPOSITORY")
        git_run_id = os.getenv("GITHUB_RUN_ID")
        git_job_name_ext = os.getenv("ETE_JOB_NAME_EXT")
        git_job_name_int = os.getenv("ETE_JOB_NAME_INT")
        assert git_job_name_ext is not None
        assert git_job_name_int is not None

        git_api_url = (
            f"https://api.github.com/repos/{git_repo}/actions/runs/{git_run_id}/jobs"
        )
        git_job_data = requests.get(git_api_url).json()

        # Get the current job dict from all job data
        current_job = None
        for job in git_job_data["jobs"]:
            if job["name"] in git_job_name_int:
                current_job = job
                break
        assert current_job is not None

        # Get total run time of job
        start_time = datetime.strptime(
            str(current_job["started_at"]), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        start_time_str = str(start_time.replace(tzinfo=None).replace(microsecond=0))
        total_time = str((datetime.now(timezone.utc) - start_time))

        if error is None:
            # If no error and is fast run, we dont care about sending error msg so give
            # _add_msg_to_artifact 'SKINNY_JOB_SUCCESS'
            if "skinny" not in git_job_name_ext:
                msg = (
                    f"*SUCCESS*, end to end run: `{git_job_name_ext}` passed all tests."
                    f"\nStart time: `{start_time_str}` UTC, total run time: `{total_time}`"
                )
                _add_msg_to_artifact(msg)
            else:
                rich_print(
                    f"SUCCESS, end to end run: {git_job_name_ext} passed all tests."
                    f"\nStart time: {start_time_str} UTC, total run time: {total_time}"
                    "\nSkipping slack message for skinny run with no errors."
                )
                _add_msg_to_artifact("SKINNY_JOB_SUCCESS")
        else:
            # Some chars in error body causing crash with env vars, remove non ascii chars
            error_body = str(error).encode("ascii", "ignore").decode("ascii")

            # Truncate error body to 500 chars
            if len(error_body) > MAX_ERROR_LENGTH:
                error_body = f"{error_body[0:MAX_ERROR_LENGTH]}..."
            error_msg = f"{type(error).__name__}: {error_body}"

            # failed_on has been set by one of the cmd runners during exception handeling to whatever cmd failed.
            msg = (
                f"*FAILURE*, end to end run: `{git_job_name_ext}` failed during: `{failed_on}` ```{error_msg}``` "
                f"Start time: `{start_time_str}` UTC, total run time: `{total_time}`"
            )
            _add_msg_to_artifact(msg)
    else:
        rich_print("Skipping slack message; not github actions run.")


def _add_msg_to_artifact(msg: str):
    """
    Gets output msg from _make_run_results_msg. Will base64 encode it,
    write to new file in artifacts folder and set env var
    ETE_JOB_FINISHED to TRUE.

    Args:
        msg : str
            The msg string to add to artifacts.

    Returns:
        None
    """

    if is_gha:
        rich_print(f"Adding to output message:\n{msg}")

        # Base64 encode output to avoid env var nonsense
        msg_bytes = msg.encode("ascii")
        msg_base64_bytes = base64.b64encode(msg_bytes)
        msg_base64_string = msg_base64_bytes.decode("ascii")

        # Make artifact folder with name ETE_ART_LOC
        artifact_folder_location = os.getenv("ETE_ART_LOC")
        artifact_job_id = os.getenv("ETE_ART_ID")
        assert artifact_folder_location is not None
        assert artifact_job_id is not None

        artifact_file_name = f"{artifact_job_id}.txt"
        artifact_path = os.path.join(artifact_folder_location, artifact_file_name)

        # Echo encoded msg into new artifact file.
        os.mkdir(artifact_folder_location)
        process = subprocess.Popen(
            f"echo {msg_base64_string} > {artifact_path}",
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
        )
        process.wait()

        # Set env var ETE_JOB_FINISHED to TRUE.
        # Workflow will construct timed out output msg for job if FALSE.
        process = subprocess.Popen(
            'echo "ETE_JOB_FINISHED=TRUE" >> "$GITHUB_ENV"',
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
        )
        process.wait()

        rich_print(
            f"ETE_JOB_FINISHED set to TRUE. Added to ETE_OUT base64 encoded message:\n{msg_base64_string}"
        )


def _send_slack_message(msg: str, ignore_invalid_response=False):
    rich_print(f"Sending msg to slack:\n{msg}")
    slack_hook = os.getenv("SLACK_HOOK_URL")
    assert slack_hook is not None
    payload = '{"text": "%s"}' % msg
    response = requests.post(slack_hook, data=payload)
    rich_print(f"Slack msg response:\n{response.text}")

    # If for some reason payload is invalid, send failure message.
    if response.text == "invalid_payload" and not ignore_invalid_response:
        _send_failure_slack_message()


def _send_failure_slack_message():
    git_repo = os.getenv("GITHUB_REPOSITORY")
    git_run_id = os.getenv("GITHUB_RUN_ID")
    git_run_url = f"https://github.com/{git_repo}/actions/runs/{git_run_id}/"
    _send_slack_message(
        f"*Finished*: {git_run_url}\n*Failed to send output for runs.*",
        ignore_invalid_response=True,
    )


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


def _cleanup_local_files(file_lists: List[str]):
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


def _get_timestamp():
    current_time = str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    return f"[bold purple][{current_time}][/bold purple]"


if __name__ == "__main__":
    try:
        t_app()
    except Exception as error:
        try:
            if is_collecting:
                # Something weird broke while running collect-and-send-logs/
                # In this case, collect-and-send-logs will make generic failed message and send to slack.
                rich_print(
                    "Something unknown has broken while running collect-and-send-logs."
                )
            else:
                # Catch any exceptions thown durring the test and make error msg for slack.
                _make_run_results_msg(error)
        except Exception as error_msger:
            # Something weird broke while running _make_run_results_msg.
            # In this case, workflow file will make generic failed message and store to artifacts.
            rich_print(
                "Something unknown has broken while running _make_run_results_msg."
            )
            raise error_msger
        else:
            raise error
