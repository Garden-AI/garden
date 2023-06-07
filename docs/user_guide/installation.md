# Installation

Installing the `garden-ai` python package contains both our SDK and our CLI. We _highly_ recommend installing the CLI with `pipx` over plain `pip`.

If you don't have `pipx` installed, please follow the instructions in the [pipx documentation](https://pypa.github.io/pipx/installation/) to install it.

Once `pipx` is installed and configured, you can install the Garden package. The simplest installation is done by:

```bash
pipx install garden-ai
pipx ensurepath
```

> [!NOTE] Note
> To be able to `import garden_ai` in a notebook or other python session, you likely need to do a plain `pip install garden-ai` with the appropriate active virtual environment. Doing so won't interfere with the `pipx`-installed CLI.


The base package includes support for sklearn by default. If you want to use Garden with PyTorch or TensorFlow, you need to specify additional "extras". For PyTorch, use:

```bash
pipx install garden-ai[pytorch]
```

For TensorFlow, use:

```bash
pipx install garden-ai[tensorflow]
```

If you want to use Garden with all the supported ML frameworks (sklearn, PyTorch, TensorFlow), you can install it with the `all` extra:

```bash
pipx install garden-ai[all]
```

This will ensure all necessary dependencies for all supported ML frameworks are installed.

To verify your installation, you can check the version of Garden:

```bash
garden-ai --version
```

You should see the version of your installed Garden package printed to the console.

Now you're ready to start Gardening!

> [!NOTE] Note
> Especially when installing with additional flavors, the CLI might be very slow to respond the first time it is invoked (or the first time the SDK is imported), but it should be much snappier every time after the first.
