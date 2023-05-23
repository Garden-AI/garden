# Contributing to Garden

We're excited you're interested in contributing to Garden! This project is all about making it easier for the scientific community to adhere to FAIR (Findable, Accessible, Interoperable, Reusable) principles for AI/ML models. Garden is an initiative by [Globus Labs](https://labs.globus.org) at UChicago and leverages Globus computing and data storage resources such as Globus Compute (formerly funcx).


## Installation

The Garden project is hosted on GitHub, and we use the `poetry` tool for package and dependency management. To get started, clone the repository and use `poetry` to install dependencies:

```bash
git clone https://github.com/Garden-AI/garden.git
cd garden
poetry install --with=test,develop
```

Note that installing the test and develop groups won't install specific ML libraries such as `tensorflow` or `pytorch`, which tend to cause problems -- install these with `poetry install --all-extras`, or specific flavors with `poetry install --extras "tensorflow"` etc.

You also might encounter problems with the `poetry.lock` file, which we keep under version control -- feel free to generate a new one with `poetry lock --no-update` and include the changes in your PR, even if `pyproject.toml` doesn't change.


## Code of Conduct

We believe in creating a welcoming and inclusive environment for all contributors. With that in mind, all contributors are expected to follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).


## Contribution Guidelines

We welcome contributions of all types and levels, whether you're fixing bugs, adding new features, or improving documentation. Here's how you can contribute:

1. Find an issue that you want to work on. If you have a new idea, please open a new issue to discuss it first.

2. Create a new branch from the `main` branch for your changes. Name it something relevant to the issue you're working on.

3. Make your changes in your new branch.

4. Once your changes are ready, open a pull request (PR). In your PR description, explain your changes and reference the issue that your PR closes.

5. Wait for a review from one of the core Garden development team members. Make any necessary changes based on their feedback.

6. Once your PR is approved, it will be merged into the `main` branch.

## Coding Standards

Garden uses standard Python coding style and conventions. We use pre-commit hooks to automatically format code with `black`. To set up the commit hooks locally, you'll need to have pre-commit [installed](https://pre-commit.com/#install) -- this should already be installed if you ran `poetry install --with=develop`, though `pre-commit` may not be on your path. Then run `pre-commit install` from the garden directory.

## Documentation
We use [mkdocs](https://www.mkdocs.org/user-guide/configuration/) to build this documentation, which should have been installed as a dependency if you ran `poetry install --with=develop`. To preview the docs locally, `poetry run mkdocs serve` (or just `mkdocs serve` in an active `poetry shell`). A documentation preview will also be linked automatically whenever you open a new PR.

### Docstrings / API documentation
We use the mkdocs extension `mkdocstrings` to parse google-style python docstrings and build some API docs automatically. The actual contents of the generated docs page is configured in the corresponding `.md` file, e.g.:

```
<!--- docs/Pipelines.md --->
::: garden_ai.pipelines
    options:
        show_root_heading: True
        show_object_full_path: True
        members:
            - Pipeline
            - RegisteredPipeline
```
The mkdocstrings options are documented [here](https://mkdocstrings.github.io/python/usage/configuration/general/).

This also means you can do a few things inside of docstrings to add some polish to the auto-generated docs:

- To link to another object's automatically generated docs from within a docstring, you can use `[link text][dotted.path.to.object]`.
- You can also use certain keywords such as `Attributes:` or `Raises:` in docstrings to generate e.g. a nicely formatted table of attributes on a class. These keywords/syntax are documented [here](https://mkdocstrings.github.io/griffe/docstrings/).
- Using the same syntax with a non-keyword (e.g. `Notes:`) will generate an admonition/callout instead.

> [!NOTE] Admonition
> You can generate admonitions from within docstrings like so:
> ```python
>  """
>  ...
>   Admonition:
>     You can generate admonitions from within docstrings like so:
>  """
> ```
>

## Testing

We use `pytest` for testing. After making changes, make sure all tests pass. You can run unit tests using the following command:

```bash
poetry run pytest -m "not integration"
```

Integration tests may be useful as a reference, but at the current stage in the project's development not likely to pass for outside contributors and are not currently part of the CI build.

**New contributions should include tests.** If you're adding a new feature, write tests that cover your feature. If you're fixing a bug, write a test that would have caught the bug.
