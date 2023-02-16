Quick Start
===========
For now, there isn't much you can do with the Garden service, but this is what
currently works.

Install Garden library and CLI
------------------------------
The garden client has been published to pypi, so the following will install the
client in your current python envioronment::

    $ pip install garden-ai

If you have it, we recommend installing the CLI with `pipx`_.::

    $ pipx install garden-ai

You should then be able to see the following: ::

    $ garden create --help

    Usage: garden create [OPTIONS] [DIRECTORY]

    Create a new Garden

    ╭─ Arguments ───────────────────────────────────────────────────────────────────╮
    │   directory      [DIRECTORY]  (Optional) if specified, this generates a       │
    │                               directory with subfolders to help organize the  │
    │                               new Garden. This is likely to be useful if you  │
    │                               want to track your Garden/Pipeline development  │
    │                               with GitHub.                                    │
    │                               [default: None]                                 │
    ╰───────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ─────────────────────────────────────────────────────────────────────╮
    │ --verbose    --no-verbose      If true, pretty-print Garden's metadata when   │
    │                                created.                                       │
    │                                [default: no-verbose]                          │
    │ --help                         Show this message and exit.                    │
    ╰───────────────────────────────────────────────────────────────────────────────╯
    ╭─ Required ────────────────────────────────────────────────────────────────────╮
    │    --author  -a      TEXT  Name an author of this Garden. Repeat this to      │
    │                            indicate multiple authors: `garden create ...      │
    │                            --author='Mendel, Gregor' --author 'Other-Author,  │
    │                            Anne' ...` (order is preserved).                   │
    │                            [default: None]                                    │
    │ *  --title   -t      TEXT  Provide an official title (as it should appear in  │
    │                            citations)                                         │
    │                            [default: None]                                    │
    │                            [required]                                         │
    │    --year    -y      TEXT  [default: 2023]                                    │
    ╰───────────────────────────────────────────────────────────────────────────────╯
    ╭─ Recommended ─────────────────────────────────────────────────────────────────╮
    │ --contributor  -c      TEXT  Acknowledge a contributor in this Garden. Repeat │
    │                              to indicate multiple (like --author).            │
    │                              [default: None]                                  │
    │ --description  -d      TEXT  A brief summary of the Garden and/or its         │
    │                              purpose, to aid discovery by other Gardeners.    │
    │                              [default: None]                                  │
    │ --tag                  TEXT  Add a tag, keyword, key phrase or other          │
    │                              classification pertaining to the Garden.         │
    │                              [default: None]                                  │
    ╰───────────────────────────────────────────────────────────────────────────────╯

.. _pipx: https://pypa.github.io/pipx/#install-pipx


Connect to Garden
-----------------
You can interact with the Garden service with the CLI. The first time you invoke
a command like `garden create`, your default browser will open a link to a
`GlobusAuth`_ generated web page. Here you can log in using your academic or
research institution's credentials; you'll be given an authorization token,
which you can paste in your terminal window. This will generate credentials in
your home directory, so you shouldn't need to repeat this step as the same user
on the same computer. Here's what that might look like: ::

    $ garden create --title=...  # etc

    Authenticating with Globus in your default web browser:

    https://auth.globus.org/v2/oauth2/authorize?client_id=...  # etc

    Please enter the code here: [your copy/pasted auth code]


You can also interact with the Garden service with a `GardenClient` instance. A trivial
application would look like::

    from garden_ai import GardenClient

    client = GardenClient()

Similarly, the first time this is run, you will need to authenticate to Garden using
a GlobusAuth flow. You will see a big URL printed in the console. Copy that link
and paste it into a browser. This will take you to the GlobusAuth where you can
log in using your academic or research institutions credentials. Upon successful
login, you will be given a token. Copy this and paste into the console to
complete login.

The Garden client will save these credentials in a file in a `.garden` directory
in your home directory. You only have to go through this process the first time
you use garden.


.. _GlobusAuth: https://www.globus.org/platform/services/auth
