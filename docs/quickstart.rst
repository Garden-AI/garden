Quick Start
===========
For now, there isn't much you can do with the Garden service, but this is what
currently works.

Install Garden Client
---------------------
The garden client has been published to pypi, so the following will install the
client in your current python envioronment::

    pip install garden-ai


Connect to Garden
--------------------------------------------
You interact with the Garden service with a `GardenClient` instance. A trivial
application would look like::

    from garden_ai import GardenClient

    client = GardenClient()

The first time this is run, you will need to authenticate to Garden using
a GlobusAuth flow. You will see a big URL printed in the console. Copy that link
and paste it into a browser. This will take you to the GlobusAuth where you can
log in using your academic or research institutions credentials. Upon successful
login, you will be given a token. Copy this and paste into the console to
complete login.

The Garden client will save these credentials in a file in a `.garden` directory
in your home directory. You only have to go through this process the first time
you use garden.
