---
name: Story
about: Suggest an idea for this project
title: My Story [Story]
labels: ''
assignees: ''

---

_Short description_

# Assumptions:
1. There will be a `GardenClient` class in the `garden` package
2. The constructor doesn't take any arguments

# Acceptance Criteria
Given I have garden-ai package installed, when I execute the following python code:
```python
from garden import GardenClient
client = GardenClient()
print(client)
```
Then I don't have any errors.

Given I push a commit to a branch that has flake8 errors, when the CI job runs, Then it reports an error and fails
