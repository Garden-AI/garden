# Frequently Asked Questions

### My entrypoint isn't working how I expect. How can I debug it?

If you're finding that an entrypoint function works correctly following a "restart kernel and run all cells" when run from within a `garden-ai notebook start` session but fails on remote execution, the `garden-ai notebook debug` command is the best place to start troubleshooting.

Garden does its best to recreate your notebook environment exactly when running remotely. But there are a few ways this process can go awry:
	- The remote execution context is a session which has been saved and reloaded. If there is a serialization or deserialization bug, the session could get corrupted.
	- Your notebook is converted to a plain python script before being executed, and there may be unexpected discrepancies between the notebook and the notebook-as-script causing problems.

The `garden-ai notebook debug` command accepts the same arguments as `garden-ai notebook start`, but instead of opening the notebook in a base image, it opens a debugging notebook in an image with your saved notebook session (i.e. the one that remote inference will use).

The debugging notebook only has a snippet of code to reload your saved session -- this is as close to the remote execution context as possible without actually executing remotely.

If calling your entrypoint function in the `garden-ai notebook debug` session behaves the same as calling your entrypoint function in a `garden-ai notebook start` session, that likely indicates a problem with the remote endpoint.

If calling your entrypoint function in the `garden-ai notebook debug` session behaves differently than in a `garden-ai notebook start` session, that indicates a problem with serializing or deserializing your notebook state. If this is the case, please open an issue on our [Github](https://github.com/Garden-AI/garden/issues), including the notebook and any additional context that might be useful so we can reproduce the bug.

### I have an M1/M2 Mac and I want to publish an entrypoint that uses tensorflow. I can't get it to work locally. How can I?

Alas, you can't. Garden aways spins up Linux containers, so you can't install the `tensorflow-macos` variant. And the normal `tensorflow` doesn't run on Apple silicon.

To work around this, we recommend using a VPS. Spin up a temporary EC2 instance (or equivalent on different cloud providers) running Linux where you can work on your notebook. When you ssh into your remote workstation, forward the port the notebook runs on so that you can still work on the notebook in your local browser. Like `ssh -L 9188:localhost:9188 my-remote-user@my-remote-workstation`.
