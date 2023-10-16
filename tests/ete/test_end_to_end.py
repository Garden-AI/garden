# type: ignore
# TODO: revisit end to end tests after publishing rework

import os
import sys
import shutil
import pickle
import jinja2
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

import numpy as np
import pandas as pd  # type: ignore

import garden_ai  # type: ignore
from garden_ai.app.main import app  # type: ignore

import globus_sdk
import globus_compute_sdk  # type: ignore

import constants as consts  # type: ignore

"""
QUICK USE:
    Local runs:
        NOTE: It is recommended you run the end to end test from the Garden poetry environment with all extras installed.

        When running the end to end test locally, there are a couple of arguments it takes
        to finetune what parts of the test you want to run.
        To begin, cd into garden/tests/ete

        Here is an basic example run:

        poetry run python3 test_end_to_end.py run-garden-end-to-end
            --garden-grant cc
            --cli-id xxx
            --cli-secret xxx
            --model-type all
            --globus-compute-endpoint default
            --use-cached-containers
            --live-print-stdout

        Let's break down what each argument is doing:
            --garden-grant cc
                Initializes the GardenClient with a cc grant and bypass the naive app login.
                If you want to run using the naive app login, use '--garden-grant at'
                NOTE: Using a cc login requires the user to provide the GARDEN_API_CLIENT_ID and GARDEN_API_CLIENT_SECRET
                NOTE: Using a cc login only allows access to the default Garden compute endpoint or any fresh endpoints created.

            --cli-id xxx
                The GARDEN_API_CLIENT_ID. Used for cc logins only (see --garden-grant).
                If this is not included during a cc run, test will prompt user for value.

            --cli-secret xxx
                The GARDEN_API_CLIENT_SECRET. Used for cc logins only (see --garden-grant).
                If this is not included during a cc run, test will prompt user for value.

            --model-type all
                The model flavors to test. Can be 'sklearn', 'sklearn-preprocessor' 'tensorflow', 'pytorch', 'custom' or 'all'.
                Will make pipeline for model type, add to new Garden, publish and test remote execution.
                If you want to test only a subset of the flavors, say tensorflow and pytorch, just use --model-type
                twice (--model-type tensorflow --model-type pytorch).
                NOTE: 'all' does not include 'custom', just 'sklearn', 'sklearn-preprocessor' 'tensorflow' and 'pytorch'
                NOTE: See below for 'custom' run example.

            --globus-compute-endpoint default
                The Globus compute endpoint to run the remote execution on.
                If value not given, will skip running remote execution test.
                If value is 'default', will test on DlHub test endpoint ('86a47061-f3d9-44f0-90dc-56ddc642c000').
                If value is 'fresh' will create a new endpoint for this run.
                NOTE: For '--garden-grant cc' runs, endpoint must be either 'fresh', 'default' or None.
                Client credential runs cannot access endpoints that other users have created.

            --use-cached-containers
                Use  previously cached containers for applicable pipelines.
                If not included, will build a new container for a pipeline every time.

            --live-print-stdout
                Print all Garden logs to stdout, in addition to the test logs.


        You can also run the test on local models. Let's say we have a sklean model that we want to test end-to-end with Garden.
        To do this, we run:

        poetry run python3 test_end_to_end.py run-garden-end-to-end
            --garden-grant cc
            --cli-id xxx
            --cli-secret xxx
            --model-type custom
            --globus-compute-endpoint fresh
            --live-print-stdout
            --custom-model-path path/to/model/model.pkl
            --custom-model-flavor sklearn
            --custom-model-reqs path/to/reqs/requirments.txt

        The test will load the model and requirements and then make a pipeline with a run_inference predict step.
        If you already have a pipeline for the model, you could instead run:

        poetry run python3 test_end_to_end.py run-garden-end-to-end
            --garden-grant cc
            --cli-id xxx
            --cli-secret xxx
            --model-type custom
            --globus-compute-endpoint fresh
            --live-print-stdout
            --custom-model-path path/to/model/model.pkl
            --custom-model-flavor sklearn
            --custom-model-pipeline path/to/pipeline/pipeline.py

        In general, the following arguments are used for testing custom model files:
            --custom-model-path xxx
                The path to a model file to run with the end-to-end test. Must be included for --model-type custom runs.

            --custom-model-flavor xxx
                The flavor of a model to run with the end-to-end test. Must be included for --model-type custom runs.

            --custom-model-pipeline xxx
                The path to a pipeline file for a custom model to run with the end-to-end test.
                Include if you already have a pipeline file for your model and don't want the test to autogenerate.

            --custom-model-reqs
                The path to the requirements file for a custom model to run with the end-to-end test.
                Include if you DON'T already have a pipeline file for your model and want the test to autogenerate.


    Github actions runs:
        The test can also be run from github actions.

        There are two different workflows that use the test:
        '.github/workflows/ete_test_skinny.yml' and '.github/workflows/ete_test_full.yml'

        The skinny test runs hourly on all supported Python versions with only sklearn model.
        It uses previously built cached containers to speed up runtime.

        The full test runs daily on py3.8.16 and tests all supported model flavors.
        It builds a new container for the pipeline.

        Both of these workflows can be run manually.

        Running these workflows on Github actions will also collect the results of the test and send
        a run summery message to the slack channel 'garden-errors'.

        RUNNING TEST WORKFLOWS FROM DIFFERENT BRANCHES:
        You can also test other branches using the end-to-end test.
        On the branch, for the workflow you want to run, change `repository-ref` on line 29 from 'main' to the name of the branch.
        Then from the actions tab in github, start the test from the workflow file on said branch.


Overview of what the test is doing:
    This script consists of two commands: run-garden-end-to-end and collect-and-send-logs

    run-garden-end-to-end is the actual end-to-end test. collect-and-send-logs is for internal github actions use only.

    The test itself is going to:
        - Test creating a new Garden.
        - Test creating a new scaffolded pipeline.
        - Test registering a new model of the flavor types selected.
        - Test creating and registering a new pipeline for each model registered in the previous step.
        - Test adding all newly registered pipelines to the previously created new Garden
        - Test publishing the new Garden
        - Test remote execution for all pipelines created in previous steps.

    Result collection:
    When running from github actions, github artifacts is used to store the test output. During the setup on github actions,
    the end-to-end test makes a new env var ETE_ART_ID and sets it to a random uuid. This uuid is also used for the name of the output file.
    After the test finishes running, _make_run_results_msg is called, which makes the output string for that run of the test.
    _make_run_results_msg then calls _add_msg_to_artifact, which base64 encodes the output msg, saves it to a new file at the location
    ${{ env.ETE_ART_LOC }}/${{ env.ETE_ART_ID }}.txt, and sets the env var ETE_JOB_FINISHED to TRUE.

    Once the workflow has finished running the end-to-end test, the value of ETE_JOB_FINISHED is checked.
    If ETE_JOB_FINISHED is FALSE, the workflow creates the file ${{ env.ETE_ART_LOC }}/${{ env.ETE_ART_ID }}.txt and echos a failed message into it.

    The workflow then uploads the folder ${{ env.ETE_ART_LOC }} as a github artifact of the name ${{ env.ETE_ART_NAME }}. ${{ env.ETE_ART_NAME }} is
    the same for all jobs in this workflow, so all job outputs will get added to this artifact.

    Once the workflow has finished all the test jobs, the workflow runs collect-and-send-logs.
    collect-and-send-logs grabs the artifact ${{ env.ETE_ART_NAME }} and reads the contents of all the files.
    At this point in the test, each job should have saved an output to a file in that artifact, so once collect-and-send-logs
    has decoded all the logs, the workflow can send the output to slack and the workflow is done.
"""


# Set to true if running collect_and_send_logs.
# Used by exception handeling to know if running test or collection.
is_collecting = False

# Set to command name that failed. Used for sending slack error message.
# If the test fails somehow without setting failed on, send unknown action as failure point.
failed_on = "unknown action"

is_gha = os.getenv("GITHUB_ACTIONS")
t_app = typer.Typer()


@t_app.command()
def run_garden_end_to_end(
    garden_grant: Optional[str] = typer.Option(
        default="cc",
        help="The grant type to initialize a GardenClient with. Must be either 'cc' for a client credential grant or 'at' for an access token grant.",
    ),
    model_type: List[str] = typer.Option(
        default=["sklearn"],
        help="A model type to test. Must be either 'sklearn', 'sklearn-preprocessor' 'tensorflow', 'pytorch', "
        "'custom' or 'all'. Include this argument multiple times to test a subset of flavors",
    ),
    use_cached_containers: Optional[bool] = typer.Option(
        default=False,
        help="If test should use container cache for a faster run. If cannot find container cache, will "
        "still build new container from scratch. Will check both local cache file and pre-defined test caches.",
    ),
    globus_compute_endpoint: Optional[str] = typer.Option(
        default=None,
        help="The globus compute endpoint to remote run the test pipelines on. If none provided, then will skip "
        "testing remote execution. If value is 'default', "
        "will test on DlHub test endpoint ('86a47061-f3d9-44f0-90dc-56ddc642c000'). "
        "If value is 'fresh' will create a new endpoint for this run.",
    ),
    live_print_stdout: Optional[bool] = typer.Option(
        default=False,
        help="If true, will print the outputs of all test commands to stdout.",
    ),
    cli_id: Optional[str] = typer.Option(
        default=None,
        help="The GARDEN_API_CLIENT_ID. Used for cc logins only (see --garden-grant). Will prompt for value if not "
        "provided and garden-grant is cc login.",
    ),
    cli_secret: Optional[str] = typer.Option(
        default=None,
        help="The GARDEN_API_CLIENT_SECRET. Used for cc logins only (see --garden-grant). "
        "Will prompt for value if not provided and garden-grant is cc login.",
    ),
    custom_model_path: Optional[str] = typer.Option(
        default=None,
        help="The path to a custom model file to run ETE test on. "
        "Used in '--model-type custom' runs only. Must be included for all '--model-type custom' runs.",
    ),
    custom_pipeline_path: Optional[str] = typer.Option(
        default=None,
        help="The path to pipeline for custom model to run ETE test on. "
        "If none provided, test will make make default pipeline instead with predict step. "
        "Used in '--model-type custom' runs only.",
    ),
    custom_model_flavor: Optional[str] = typer.Option(
        default=None,
        help="The flavor of custom model to run ETE test on. "
        "Must be either 'sklearn', 'tensorflow', or 'pytorch' Used in '--model-type custom' runs only. "
        "Must be included for all '--model-type custom' runs.",
    ),
    custom_model_reqs: Optional[str] = typer.Option(
        default=None,
        help="The path to custom models requirements file. Not needed if '--custom-pipeline-path' is given. "
        "Used in '--model-type custom' runs only.",
    ),
):
    """See 'QUICK USE' at the top of test_end_to_end.py for more info on how to use."""
    """
    Tests garden commands: 'garden create', 'garden add-pipeline', 'garden search', 'garden publish',
    'model register', 'pipeline create', 'pipeline register' and remote execution of gardens.

    Args:
        garden_grant : Optional[str]
            The globus auth grant to initilize the GardenClient.
            Can be either 'at' (access token) to login with a naive app login or
            'cc' (client credentials) to login with a client credientials login.
        model_type : Optional[List[str]]
            The model type with which to test Garden. Can be either 'sklearn', 'sklearn-preprocessor',
            'tensorflow', 'pytorch', 'custom' or 'all'.
            Can include argument multiple times to test subset of all.
        use_cached_containers : Optional[bool]
            If set to true, checks cache for all needed containers and skips building where possible.
            If no cache found, will still build containers.
        globus_compute_endpoint : Optional[str]
            The globus compute endpoint to remote run the test pipelines on.
            If none provided, then will skip testing remote execution.
            If value is 'default', will test on DlHub test endpoint ('86a47061-f3d9-44f0-90dc-56ddc642c000').
            If value is 'fresh' will create a new endpoint for this run.
            For '--garden-grant cc' runs, endpoint must be either 'fresh', 'default' or None.
        live_print_stdout : Optional[bool]
            If set to true, will print all CLI outputs to stdout.
        cli_id : Optional[str]
            The GARDEN_API_CLIENT_ID. Only for cc local runs (see --garden-grant).
            Will prompt for value if not provided and cc login.
        cli_secret : Optional[str]
            The GARDEN_API_CLIENT_SECRET. Only for cc local runs (see --garden-grant).
            Will prompt for value if not provided and cc login.
        custom_model_path : Optional[str]
            Used for '--model-type custom' runs only.
            The path to a model file to run with the end to end test.
            Must be included for --model-type custom runs.
        custom_model_flavor : Optional[str]
            Used for '--model-type custom' runs only.
            The flavor of a model to run with the end to end test.
            Must be included for --model-type custom runs.
        custom_model_pipeline
            Used for '--model-type custom' runs only.
            The path to a pipeline file for a custom model to run with the end to end test.
            Include if you already have a pipeline file for your model and dont want the test to autogenerate.
        custom_model_reqs
            Used for '--model-type custom' runs only.
            The path to the requirments file for a custom model to run with the end to end test.
            Include if you don't already have a pipeline file for your model and want the test to autogenerate.

    Returns:
        None
    """

    # Set up ETE test
    rich_print("\n[bold blue]Setup ETE Test[/bold blue]\n")

    constants = consts.ETEConstants()

    rich_print(f"Garden grant type set to: [blue]{garden_grant}[/blue]\n")

    # Set what model types are being run
    run_sklearn = False
    run_sklearn_pre = False
    run_tf = False
    run_torch = False
    run_custom = False
    for mt in model_type:
        if (
            mt not in garden_ai.mlmodel.ModelFlavor._value2member_map_
            and mt != "sklearn-preprocessor"
            and mt != "all"
            and mt != "custom"
        ):
            raise Exception(f"{mt} is not a valid model-type.")
        if mt == "all":
            run_sklearn = True
            run_sklearn_pre = True
            run_tf = True
            run_torch = True
        elif mt == "sklearn":
            run_sklearn = True
        elif mt == "sklearn-preprocessor":
            run_sklearn_pre = True
        elif mt == "tensorflow":
            run_tf = True
        elif mt == "pytorch":
            run_torch = True
        elif mt == "custom":
            run_custom = True

            assert custom_model_path is not None
            assert custom_model_flavor is not None

            if (
                custom_model_flavor
                not in garden_ai.mlmodel.ModelFlavor._value2member_map_
            ):
                raise Exception(
                    f"{custom_model_flavor} is not a valid flavor. custom-model-flavor must be a flavor supported by Garden."
                )

            constants.custom_model_location = custom_model_path
            constants.custom_model_flavor = custom_model_flavor

            if custom_pipeline_path is not None:
                constants.custom_pipeline_path = custom_pipeline_path
                constants.custom_make_new_pipeline = True
            else:
                assert custom_model_reqs is not None
                constants.custom_model_reqs = custom_model_reqs

    rich_print(f"Testing with [blue]sklearn[/blue] model: {run_sklearn}")
    rich_print(
        f"Testing with [blue]sklearn preprocessor[/blue] model: {run_sklearn_pre}"
    )
    rich_print(f"Testing with [blue]tensorflow[/blue] model: {run_tf}")
    rich_print(f"Testing with [blue]pytorch[/blue] model: {run_torch}")
    rich_print(f"Testing with [blue]custom[/blue] model: {run_custom}\n")
    if run_custom:
        rich_print(
            f"Custom model is type [blue]{constants.custom_model_flavor}[/blue]\n"
        )

    # If run with --live-print-stdout, will print all commands output to console.
    runner = None
    rich_print(f"CliRunner live print set to: [blue]{live_print_stdout}[/blue]")
    if live_print_stdout:
        runner = _make_live_print_runner()
    else:
        runner = CliRunner()

    rich_print(f"Used container cache set to: [blue]{use_cached_containers}[/blue]")
    rich_print(
        f"Globus Compute Endpoint set to: [blue]{globus_compute_endpoint}[/blue]"
    )

    # Cleanup any left over files generated from the test
    _cleanup_local_files(constants.local_files_list)

    rich_print(f"\ngarden_ai module location: {garden_ai.__file__}")

    rich_print("\n[bold blue]Starting ETE Test[/bold blue]\n")

    # Change working dir to .garden
    old_cwd = os.getcwd()
    os.chdir(constants.key_store_path)

    client = None
    if garden_grant == "cc":
        # Create GardenClient with ClientCredentialsAuthorizer
        if is_gha:
            GARDEN_API_CLIENT_ID = os.getenv("GARDEN_API_CLIENT_ID", None)
            GARDEN_API_CLIENT_SECRET = os.getenv("GARDEN_API_CLIENT_SECRET", None)
            assert (
                GARDEN_API_CLIENT_ID is not None
                and GARDEN_API_CLIENT_SECRET is not None
            )
        else:
            if cli_id is not None and cli_secret is not None:
                GARDEN_API_CLIENT_ID = cli_id
                GARDEN_API_CLIENT_SECRET = cli_secret
            else:
                # Prompt for CC login secrets during local run.
                # Only if --cli-id and --cli-secret not given.
                GARDEN_API_CLIENT_ID = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_ID here: "
                ).strip()
                GARDEN_API_CLIENT_SECRET = Prompt.ask(
                    "Please enter the GARDEN_API_CLIENT_SECRET here: "
                ).strip()

        # CC login with FUNCX
        os.environ["FUNCX_SDK_CLIENT_ID"] = GARDEN_API_CLIENT_ID
        os.environ["FUNCX_SDK_CLIENT_SECRET"] = GARDEN_API_CLIENT_SECRET

        if (
            globus_compute_endpoint is not None
            and globus_compute_endpoint != constants.default_endpoint
            and globus_compute_endpoint != "default"
            and globus_compute_endpoint != "fresh"
        ):
            raise Exception(
                f"Invalid globus compute endpoint; For client credential runs, compute endpoint must be either "
                f"'{constants.default_endpoint}', 'default', 'fresh' or None."
            )

        client = _make_garden_client_with_cc(
            GARDEN_API_CLIENT_ID, GARDEN_API_CLIENT_SECRET
        )

    elif garden_grant == "at":
        # Create GardenClient normally with access token grant
        GARDEN_API_CLIENT_ID = "none"
        GARDEN_API_CLIENT_SECRET = "none"
        constants.pipeline_template_name = "ete_pipeline_at"
        client = _make_garden_client_with_at()
    else:
        raise Exception(
            "Invalid garden grant type; must be either cc (Client credential grant) or at (Access token grant)."
        )

    # Patch all instances of GardenClient with our new grant type one and run tests with patches.
    # If use cached containers is true then also patch _read_local_cache method.
    if use_cached_containers:
        old_cache = garden_ai.local_data._read_local_cache()

        with mocker.patch(
            "garden_ai.app.garden.GardenClient"
        ) as mock_garden_gc, mocker.patch(
            "garden_ai.app.model.GardenClient"
        ) as mock_model_gc, mocker.patch(
            "garden_ai.app.pipeline.GardenClient"
        ) as mock_pipeline_gc, mocker.patch(
            "garden_ai.client._read_local_cache"
        ) as mock_read_cache:
            mock_garden_gc.return_value = client
            mock_model_gc.return_value = client
            mock_pipeline_gc.return_value = client

            # add pipeline container cache entries to local cache and mocks new cache to _read_local_cache
            for key, value in old_cache.items():
                if key not in constants.mock_container_cache:
                    constants.mock_container_cache[key] = value
            mock_read_cache.return_value = constants.mock_container_cache

            _run_test_cmds(
                client,
                runner,
                globus_compute_endpoint,
                run_sklearn,
                run_sklearn_pre,
                run_tf,
                run_torch,
                run_custom,
                GARDEN_API_CLIENT_ID,
                GARDEN_API_CLIENT_SECRET,
                constants,
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
                run_sklearn_pre,
                run_tf,
                run_torch,
                run_custom,
                GARDEN_API_CLIENT_ID,
                GARDEN_API_CLIENT_SECRET,
                constants,
            )

    rich_print("\n[bold blue]Finished ETE Test successfully; cleaning up[/bold blue]\n")

    os.chdir(old_cwd)

    # Cleanup local files
    _cleanup_local_files(constants.local_files_list)

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
    globus_compute_endpoint: Optional[str],
    run_sklearn: bool,
    run_sklearn_pre: bool,
    run_tf: bool,
    run_torch: bool,
    run_custom: bool,
    GARDEN_API_CLIENT_ID: str,
    GARDEN_API_CLIENT_SECRET: str,
    constants: consts.ETEConstants,
):
    """
    Runs all garden CLI commands after env has been setup.

    Args:
        client : garden_ai.GardenClient
            The GardenClient to use.
        runner : CliRunner
            The CliRunner to run the garden commands with.
        globus_compute_endpoint : Optional[str]
            The globus compute endpoint to run remote execution test on. If None will not run remote execution.
        run_sklearn : bool
            Whether to run sklearn model tests.
        run_sklearn_pre : bool
            Whether to run sklearn preprocessor tests.
        run_tf : bool
            Whether to run tensorflow model tests.
        run_torch : bool
            Whether to run pytorch model tests.
        run_custom : bool
            Whether to run custom model tests.
        GARDEN_API_CLIENT_ID : str
            If run is client credential grant, is set to secret GARDEN_API_CLIENT_ID, otherwise is 'none'
        GARDEN_API_CLIENT_SECRET : str
            If run is client credential grant, is set to secret GARDEN_API_CLIENT_SECRET, otherwise is 'none'
        constants : ETEConstants
            All constants for test run, loaded from constants.py

    Returns:
        None
    """

    # Garden create
    new_garden = _test_garden_create(
        constants.example_garden_data, constants.garden_title, runner
    )

    # Pipeline create
    _test_pipeline_create(
        constants.example_pipeline_data,
        constants.key_store_path,
        constants.scaffolded_pipeline_folder_name,
        runner,
    )

    if run_sklearn:
        # Model register sklearn
        sklearn_model_full_name = _test_model_register(
            constants.sklearn_model_location,
            "sklearn",
            constants.sklearn_model_name,
            constants.sklearn_serialize,
            runner,
        )
        # Pipeline make sklearn
        sklearn_pipeline_local = _make_pipeline_file(
            constants.sklearn_pipeline_name,
            sklearn_model_full_name,
            constants.sklearn_model_reqs_location,
            constants.sklearn_pipeline_path,
            constants.pipeline_template_name,
            constants.pipeline_template_location,
            constants.sklearn_func,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register sklearn
        sklearn_pipeline = _test_pipeline_register(
            constants.sklearn_pipeline_path,
            sklearn_pipeline_local,
            sklearn_model_full_name,
            "sklearn",
            runner,
        )
        # Add sklearn pipeline to garden
        _test_garden_add_pipeline(new_garden, sklearn_pipeline, runner)

    if run_sklearn_pre:
        # Model register sklearn preprocessor
        sklearn_pre_model_full_name = _test_model_register(
            constants.sklearn_pre_model_location,
            "sklearn",
            constants.sklearn_pre_model_name,
            constants.sklearn_serialize,
            runner,
        )
        # Pipeline make sklearn preprocessor
        sklearn_pre_pipeline_local = _make_pipeline_file(
            constants.sklearn_pre_pipeline_name,
            sklearn_pre_model_full_name,
            constants.sklearn_model_reqs_location,
            constants.sklearn_pre_pipeline_path,
            constants.pipeline_template_name,
            constants.pipeline_template_location,
            constants.sklearn_pre_func,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register sklearn preprocessor
        sklearn_pre_pipeline = _test_pipeline_register(
            constants.sklearn_pre_pipeline_path,
            sklearn_pre_pipeline_local,
            sklearn_pre_model_full_name,
            "sklearn",
            runner,
        )
        # Add sklearn preprocessor pipeline to garden
        _test_garden_add_pipeline(new_garden, sklearn_pre_pipeline, runner)

    if run_tf:
        # Model register tensorflow
        tf_model_full_name = _test_model_register(
            constants.tf_model_location,
            "tensorflow",
            constants.tf_model_name,
            constants.keras_serialize,
            runner,
        )
        # Pipeline make tensorflow
        tf_pipeline_local = _make_pipeline_file(
            constants.tf_pipeline_name,
            tf_model_full_name,
            constants.tf_model_reqs_location,
            constants.tf_pipeline_path,
            constants.pipeline_template_name,
            constants.pipeline_template_location,
            constants.tf_func,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )
        # Pipeline register tensorflow
        tf_pipeline = _test_pipeline_register(
            constants.tf_pipeline_path,
            tf_pipeline_local,
            tf_model_full_name,
            "tensorflow",
            runner,
        )
        # Add tensorflow pipeline to garden
        _test_garden_add_pipeline(new_garden, tf_pipeline, runner)

    if run_torch:
        # Use torch specific pipeline to call model.eval() and torch.no_grad()
        if constants.pipeline_template_name == "ete_pipeline_at":
            constants.pipeline_template_name = "ete_pipeline_torch_at"
        if constants.pipeline_template_name == "ete_pipeline_cc":
            constants.pipeline_template_name = "ete_pipeline_torch_cc"

        # Model register pytorch
        torch_model_full_name = _test_model_register(
            constants.torch_model_location,
            "pytorch",
            constants.torch_model_name,
            constants.torch_serialize,
            runner,
        )
        # Pipeline make pytorch
        torch_pipeline_local = _make_pipeline_file(
            constants.torch_pipeline_name,
            torch_model_full_name,
            constants.torch_model_reqs_location,
            constants.torch_pipeline_path,
            constants.pipeline_template_name,
            constants.pipeline_template_location,
            constants.torch_func,
            GARDEN_API_CLIENT_ID,
            GARDEN_API_CLIENT_SECRET,
            client,
        )

        # Pipeline register pytorch
        torch_pipeline = _test_pipeline_register(
            constants.torch_pipeline_path,
            torch_pipeline_local,
            torch_model_full_name,
            "pytorch",
            runner,
        )
        # Add pytorch pipeline to garden
        _test_garden_add_pipeline(new_garden, torch_pipeline, runner)

    if run_custom:
        # Model register custom
        custom_model_full_name = _test_model_register(
            constants.custom_model_location,
            constants.custom_model_flavor,
            constants.custom_model_name,
            constants.custom_serialize,
            runner,
        )

        if constants.custom_make_new_pipeline:
            # Pipeline make custom
            custom_pipeline_local = _make_pipeline_file(
                constants.custom_pipeline_name,
                custom_model_full_name,
                constants.custom_model_reqs,
                constants.custom_pipeline_path,
                constants.pipeline_template_name,
                constants.pipeline_template_location,
                constants.custom_func,
                GARDEN_API_CLIENT_ID,
                GARDEN_API_CLIENT_SECRET,
                client,
            )
        else:
            # User provided a pipeline file, dont need to make, but still need dummy pipeline object for _test_pipeline_register.
            @garden_ai.step
            def run_inference(arg: object) -> object:
                """placeholder"""
                return arg

            custom_pipeline_local = client.create_pipeline(
                title=constants.custom_pipeline_name,
                authors=["ETE Test Author"],
                contributors=[],
                steps=[run_inference],  # type: ignore
                tags=[],
                description="ETE Test Pipeline",
                year=str(datetime.now().year),
                python_version=f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
            )

        # Pipeline register custom
        custom_pipeline = _test_pipeline_register(
            constants.custom_pipeline_path,
            custom_pipeline_local,
            custom_model_full_name,
            constants.custom_model_flavor,
            runner,
        )

        # Add custom pipeline to garden
        _test_garden_add_pipeline(new_garden, custom_pipeline, runner)

    # Publish the garden
    published_garden = _test_garden_publish(new_garden, runner)

    # Search for our garden
    _test_garden_search(published_garden, runner)

    # Test run all selected pipelines on globus compute endpoint
    if globus_compute_endpoint is not None:
        # remove typer optional type
        assert globus_compute_endpoint is not None
        if run_sklearn:
            _test_run_garden_on_endpoint(
                published_garden,
                constants.sklearn_pipeline_name,
                constants.sklearn_input_data_location,
                constants.sklearn_expected_data_location,
                globus_compute_endpoint,
                constants,
                client,
            )

        if run_sklearn_pre:
            _test_run_garden_on_endpoint(
                published_garden,
                constants.sklearn_pre_pipeline_name,
                constants.sklearn_pre_input_data_location,
                constants.sklearn_pre_expected_data_location,
                globus_compute_endpoint,
                constants,
                client,
            )

        if run_tf:
            _test_run_garden_on_endpoint(
                published_garden,
                constants.tf_pipeline_name,
                constants.tf_input_data_location,
                constants.tf_expected_data_location,
                globus_compute_endpoint,
                constants,
                client,
            )
        if run_torch:
            _test_run_garden_on_endpoint(
                published_garden,
                constants.torch_pipeline_name,
                constants.torch_input_data_location,
                constants.torch_expected_data_location,
                globus_compute_endpoint,
                constants,
                client,
            )
        if run_custom:
            rich_print(
                "Skipping remote execution for custom model; not yet implemented"
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
        assert gardens_after is not None
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
        assert garden_after_addition is not None
        local_pipelines = garden_ai.local_data.get_all_local_pipelines()
        assert local_pipelines is not None
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
    model_location: str,
    flavor: str,
    short_name: str,
    serialize_type: Optional[str],
    runner: CliRunner,
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
        serialize_type : str
            The serialization type for the model
        runner : CliRunner
            The CliRunner to run the model commands with.

    Returns: str
        The full name of the registerd model.
    """

    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]model register[/italic red] using model flavor: [blue]{flavor}[/blue]"
        )

        if serialize_type is not None:
            command = [
                "model",
                "register",
                short_name,
                str(model_location),
                flavor,
                "-s",
                serialize_type,
            ]
        else:
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
        assert local_models is not None

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
        assert local_pipelines is not None

        registered_pipeline = None
        for local_pipeline in local_pipelines:
            if str(pipeline.doi) == str(local_pipeline.doi):
                registered_pipeline = local_pipeline
                break

        assert registered_pipeline is not None
        assert pipeline.title == registered_pipeline.title
        assert registered_pipeline.doi is not None
        assert registered_pipeline.steps is not None

        assert len(registered_pipeline.steps) == 1
        assert registered_pipeline.steps[0]["model_full_names"] is not None
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
    expected_data_file: str,
    globus_compute_endpoint: str,
    constants: consts.ETEConstants,
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
        expected_data_file : str
            The path to the expected model output data to check results against for this remote execution.
        globus_compute_endpoint : str
            The the globus compute endpoint to run on.
        constants : consts.ETEConstants,
            All constants for test run, loaded from constants.py
        client : garden_ai.GardenClient,
            The GardenClient to run the remote exectution with.

    Returns:
        None
    """
    is_fresh_endpoint = False

    try:
        rich_print(
            f"\n{_get_timestamp()} Starting test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}[/blue]"
        )

        # This needs to run after setting FUNCX_SDK env vars so fresh endpoint is registered to cc login user
        if globus_compute_endpoint == "default":
            globus_compute_endpoint = constants.default_endpoint
        elif globus_compute_endpoint == "fresh":
            is_fresh_endpoint = True
            globus_compute_endpoint = _make_compute_endpoint(
                constants.fresh_endpoint_name
            )

        with open(input_data_file, "rb") as f:
            Xtest = pickle.load(f)

        test_garden = client.get_published_garden(garden.doi)

        run_pipeline = getattr(test_garden, pipeline_name)
        result = run_pipeline(Xtest, endpoint=globus_compute_endpoint)

        assert result is not None

        rich_print(f"Result: \n{result}")

        with open(expected_data_file, "rb") as f:
            result_expected = pickle.load(f)

        if isinstance(result, np.ndarray):
            assert np.array_equal(result, result_expected)
        elif isinstance(result, pd.DataFrame):
            assert result.equals(result_expected)
        else:
            import torch  # type: ignore

            if isinstance(result, torch.Tensor):
                assert torch.allclose(result, result_expected)
            else:
                assert result == result_expected

        if is_fresh_endpoint:
            _delete_compute_endpoint(constants.fresh_endpoint_name)

        rich_print(
            f"{_get_timestamp()} Finished test: [italic red]garden remote execution[/italic red] using pipeline: [blue]{pipeline_name}"
            "[/blue] with no errors"
        )
    except Exception as error:
        global failed_on
        failed_on = "run garden on remote endpoint"
        rich_print(
            f"{_get_timestamp()} Failed test: [italic red]run garden remote[/italic red] using pipeline: [blue]{pipeline_name}[/blue]"
        )

        if is_fresh_endpoint:
            _delete_compute_endpoint(constants.fresh_endpoint_name)

        raise error


def _make_pipeline_file(
    short_name: str,
    model_full_name: str,
    req_file_path: str,
    save_path: str,
    template_name: str,
    pipeline_template_location: str,
    pipeline_model_func: str,
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
        pipeline_model_func: str
            The model function to call in run interface.
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
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
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
            pipeline_model_func=pipeline_model_func,
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


def _make_compute_endpoint(endpoint_name):
    rich_print(
        f"{_get_timestamp()} Making fresh compute endpoint [blue]{endpoint_name}[/blue]."
    )
    process = subprocess.Popen(
        f"globus-compute-endpoint configure {endpoint_name}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
    )
    process.wait()

    process = subprocess.Popen(
        f"globus-compute-endpoint start {endpoint_name}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
    )
    process.wait()

    compute_cli = globus_compute_sdk.Client()
    all_endpoints = compute_cli.get_endpoints()
    for ep in all_endpoints:
        if ep["name"] == endpoint_name:
            ep_id = ep["uuid"]
            rich_print(
                f"{_get_timestamp()} Finished making fresh compute endpoint [blue]{endpoint_name}[/blue] with uuid [blue]{ep_id}[/blue]."
            )
            return ep_id
    raise Exception("Unable to make new fresh globus compute endpoint.")


def _delete_compute_endpoint(endpoint_name):
    rich_print(
        f"{_get_timestamp()} Deleting fresh compute endpoint [blue]{endpoint_name}[/blue]."
    )
    process = subprocess.Popen(
        f"globus-compute-endpoint stop {endpoint_name}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
    )
    process.wait()

    process = subprocess.Popen(
        f"yes y | globus-compute-endpoint delete {endpoint_name}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
    )
    process.wait()

    rich_print(
        f"{_get_timestamp()} Finished deleting fresh compute endpoint [blue]{endpoint_name}[/blue]."
    )


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
