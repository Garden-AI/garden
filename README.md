# 🌱 Garden: FAIR AI/ML Model Publishing Framework

[![NSF-2209892](https://img.shields.io/badge/NSF-2209892-blue)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2209892&HistoricalAwards=false)
[![PyPI](https://badge.fury.io/py/garden-ai.svg)](https://badge.fury.io/py/garden-ai)
[![Tests](https://github.com/Garden-AI/garden/actions/workflows/pypi.yaml/badge.svg)](https://github.com/Garden-AI/garden/actions/workflows/pypi.yaml)
[![tests](https://github.com/Garden-AI/garden/actions/workflows/ci.yaml/badge.svg)](https://github.com/Garden-AI/garden/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/Garden-AI/garden/graph/badge.svg?token=WYINAGF0S4)](https://codecov.io/github/Garden-AI/garden)

## At a Glance:

- **Easy Model Publishing**: Publish pre-trained AI/ML models from a notebook with just a few commands
- **Reproducible Environments**: Use containers to ensure consistent execution across different systems
- **Remote Execution**: Run your models (or others) remotely on HPC resources seamlessly
- **Discoverable Collections**: Organize models into "Gardens" for easy discovery and comparison
- **Metadata Management**:  Capture and manage metadata of related datasets, papers, or code repositories for better searchability

## Why Garden?

Garden addresses key challenges faced by academic researchers in discovering, reproducing, and running AI/ML models:

1. **Reproducibility**: Garden eliminates environment inconsistencies by containerizing models, ensuring they run consistently across different systems.

2. **Discoverability**: With curated "Gardens" of models, researchers can easily find, compare, and curate relevant models for their work.

3. **Accessibility**: Garden simplifies the process of running models on diverse computing resources, from local machines to HPC clusters, via Globus Compute integration.

4. **Time-saving**: By handling environment management and system-specific quirks, Garden significantly reduces the time researchers spend on setup and configuration.

5. **Collaboration**: FAIR principles (Findable, Accessible, Interoperable, Reusable) and standardized publishing make it easier for researchers to share their work and build upon others' contributions.

Garden aims to let researchers focus on their science, not on the intricacies of software environments and computing infrastructure.

### What's a Garden?

A "Garden" is a citable collection of published pre-trained AI/ML models.


## Quick Start

1. Install the garden CLI:

    ``` sh
    pipx install garden-ai
    ```

## Documentation

For more documentation, including installation instructions, tutorials, and API references, see our [latest docs](https://garden-ai.readthedocs.io/).

## Contributing

We welcome contributions from the community! Please see our [Contributing Guide](https://garden-ai.readthedocs.io/en/latest/developer_guide/contributing/) for more information on how to get started.

## Support
This work was supported by the National Science Foundation under NSF Award Number: 2209892 "Frameworks: Garden: A FAIR Framework for Publishing and Applying AI Models for Translational Research in Science, Engineering, Education, and Industry".
