import os
import typer
import requests
from datetime import datetime, timezone, timedelta

t_app = typer.Typer()


@t_app.command()
def report_timeout_jobs(
    step_name: str = typer.Option(
        default="none",
        help="The name of the step to check.",
    ),
    timeout: int = typer.Option(
        default=90,
        help="The timeout amount.",
    ),
):
    is_gha = os.getenv("GITHUB_ACTIONS")
    if not is_gha:
        raise Exception("For github actions use only.")
    if step_name == "none":
        raise Exception("Must set --step-name.")

    slack_hook = os.getenv("SLACK_HOOK_URL")
    git_repo = os.getenv("GITHUB_REPOSITORY")
    git_run_id = os.getenv("GITHUB_RUN_ID")
    git_job_name = os.getenv("GITHUB_JOB_NAME")

    git_api_url = (
        f"https://api.github.com/repos/{git_repo}/actions/runs/{git_run_id}/jobs"
    )

    git_jobs_url = f"https://github.com/{git_repo}/actions/runs/{git_run_id}/"

    git_workflow_data = requests.get(git_api_url).json()

    failed_job = False
    for job in git_workflow_data["jobs"]:
        if job["name"] in git_job_name:
            for step in job["steps"]:
                if step_name == step["name"]:
                    start_time = datetime.strptime(
                        str(step["started_at"]), "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).replace(tzinfo=timezone.utc)
                    end_time = datetime.strptime(
                        str(step["completed_at"]), "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).replace(tzinfo=timezone.utc)
                    total_time = end_time - start_time
                    if total_time >= timedelta(minutes=timeout):
                        failed_job = True
                        break
        if failed_job:
            break

    if failed_job:
        msg = f"*FAILURE*, end to end run: `{str(git_job_name)}` timed out after {timeout} minutes.\n"
        msg += f"See Github actions run for more information:\n{git_jobs_url}"
        _send_slack_message(msg, slack_hook)
    else:
        print(f"Job {git_job_name} did not time out.")


def _send_slack_message(msg, slack_hook):
    payload = '{"text": "%s"}' % msg
    requests.post(slack_hook, data=payload)


if __name__ == "__main__":
    t_app()
