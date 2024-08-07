# 🌱 Garden: FAIR AI/ML Model Publishing Framework

[![NSF-2209892](https://img.shields.io/badge/NSF-2209892-blue)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2209892&HistoricalAwards=false)
[![PyPI](https://badge.fury.io/py/garden-ai.svg)](https://badge.fury.io/py/garden-ai)
[![Tests](https://github.com/Garden-AI/garden/actions/workflows/pypi.yaml/badge.svg)](https://github.com/Garden-AI/garden/actions/workflows/pypi.yaml)
[![tests](https://github.com/Garden-AI/garden/actions/workflows/ci.yaml/badge.svg)](https://github.com/Garden-AI/garden/actions/workflows/ci.yaml)

Garden is a framework designed to make publishing and applying AI/ML models for translational research in science, engineering, education, and industry easier and more accessible. By adhering to FAIR (Findable, Accessible, Interoperable, Reusable) principles, Garden aims to enhance collaboration and reproducibility in the scientific community.

## At a Glance:

- **Easy Model Publishing**: Publish your AI/ML models with just a few commands
- **Reproducible Environments**: Use containers to ensure consistent execution across different systems (via globus compute)
- **Remote Execution**: Run your models (or others') remotely on HPC resources seamlessly
- **Discoverable Collections**: Organize models into "Gardens" for easy discovery and comparison
- **Metadata Management**: Automatically capture and manage metadata for better searchability

## Quick Start

1. Install Garden:

``` sh
pipx install garden-ai
```
2. Set up Docker on your system (required for local development and testing)

3. Create your first Garden:

``` sh
garden-ai garden create --title "My First Garden" --author "Your Name"
```

4. Start a notebook in an isolated environment:

``` sh
garden-ai notebook start my_model.ipynb --base-image=3.10-sklearn
```
5. Define a function invoking your model in the notebook and publish it:

``` sh
garden-ai notebook publish my_model.ipynb
```

For a more detailed walkthrough, check out our [15-minute tutorial](https://garden-ai.readthedocs.io/en/latest/user_guide/tutorial/).

## Documentation

For more documentation, including installation instructions, tutorials, and API references, visit our [Read the Docs site](https://garden-ai.readthedocs.io/).

## Contributing

We welcome contributions from the community! Please see our [Contributing Guide](https://garden-ai.readthedocs.io/en/latest/developer_guide/contributing/) for more information on how to get started.

## Support
This work was supported by the National Science Foundation under NSF Award Number: 2209892 "Frameworks: Garden: A FAIR Framework for Publishing and Applying AI Models for Translational Research in Science, Engineering, Education, and Industry".
