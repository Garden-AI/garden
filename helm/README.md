Note: These instructions paraphrase/revise the instructions from the Globus Compute [kubernetes endpoint chart](https://github.com/funcx-faas/funcX/blob/main/helm/README.md) as well as some old DLHub deployment docs.

# Prerequisites
- `kubectl` installed, configured per instructions [here](https://login-river.ssl-hep.org/login/river) (requires login).
  - sanity check: `kubectl -n dlhub get pods`. If you get something like `Error from server (Forbidden): pods is forbidden: User "owenpriceskelly@uchicago.edu" cannot list resource "pods" in API group "" in the namespace “dlhub”` you might need to email Lincoln Bryant to get permissions sorted out.
- `helm` installed. note that helm commands also need to be namespaced with `-n dlhub`.

# Kubernetes Endpoint Helm Chart
This chart will deploy a functioning Kubernetes endpoint into your cluster. It
will launch workers with a specified container image into a namespace.

It uses Role Based Access Control to create a service account and assign it
permissions to create the worker pod.

## How to Use
There are two required values to specify in the `values.yaml` file:
`endpointUUID` and authentication.

### Authentication
Under the hood, the Globus Compute Endpoint uses the Globus Compute SDK for
communication with the web services, which requires an authenticated user for
most API routes.  The Globus Compute SDK can use either client credentials or
user credentials, both of which this chart implements.  The next two sections
describe how to implement each.

NOTE: Garden Demo Endpoint is currently deployed using Client Credentials for authorization.

#### Client Credentials
The Globus Compute SDK supports use of Globus Auth Client Credentials.  In practice,
that means exporting two variables into the endpoint's environment:

* `FUNCX_SDK_CLIENT_ID`
* `FUNCX_SDK_CLIENT_SECRET`

These variables may be generated by following the steps in the [Registering an
Application](https://docs.globus.org/api/auth/developer-guide/#register-app)
section on the [Globus Auth Developer's
Guide](https://docs.globus.org/api/auth/developer-guide/).

Outside of this chart, use of client credentials is also documented for [normal
Globus Compute SDK
usage](https://funcx.readthedocs.io/en/latest/sdk.html#client-credentials-with-globus-compute-clients).

Add these variables to a secret object in Kubernetes.  For example, to put them
into a Kubernetes store named `my-secrets`, you could create a temporary env file
and load them:

```
$ (umask 077; touch client_creds.env)  # create with 0600 (-rw-------) perms
$ cat > client_creds.env
FUNCX_SDK_CLIENT_ID=11111111-2222-4444-8888-000000000000
FUNCX_SDK_CLIENT_SECRET=yoursecret
^D
$ kubectl create secret generic my-secrets --from-env-file ./client_creds.env
```

Then, specify the secret name in the values file, and tell the chart to use
the client credentials:
```
secrets: my-secrets
useClientCredentials: true
```

#### User Credentials
If you have previously utilized the Globus Compute client, then jump to step 2 as you
will already have generated the credential file.

1. Instantiate a client with these two commands:
    ```shell
    $ pip install globus-compute-sdk
    $ python -c "from globus_compute_sdk import Client; Client()"
    ```
    A prompt beginning "Please authenticate with Globus here:" and with a long
    URL to a Globus authentication workflow will print to the terminal; follow
    the URL and paste the resulting token back into the terminal.  This will
    create the credential file at `$HOME/.globus_compute/storage.db`.
1. Create a Kubernetes secret named `compute-sdk-tokens` with this file:
    ```shell
    $ kubectl create secret generic compute-sdk-tokens --from-file=$HOME/.globus_compute/storage.db
    ```
    It is important to name the secret `compute-sdk-tokens` as this chart looks
    for that secret name, specifically.

With the `compute-sdk-tokens` Kubernetes secret object created, tell the chart to
use the user credentials:

```
useUserCredentials: true
```

### assigning the UUID

Generating the UUID needs a bit of a hack. The endpoint has to have already been registered with the Globus Compute Service before it can be (re-)deployed here. This process is a little different depending on how you set up credentials, but the gist is:
- make a new endpoint locally
- write down its uuid for `values.yaml`
- stop/delete the endpoint locally

If deploying using your own user credentials, you shouldn't need to do anything special (since those credentials already "own" the endpoint uuid):

``` sh
$ globus-compute-endpoint configure my-endpoint
Created profile for endpoint named <my-endpoint>
 ...
$ globus-compute-endpoint start my-endpoint
Starting endpoint; registered ID: e81e502f-e19b-4f81-a09d-1fc5c104b153

$ globus-compute-endpoint stop my-endpoint
> Endpoint <my-endpoint> is now stopped
```

If you're using client credentials, however, you need to make sure that when you first configure the endpoint, the `FUNCX_SDK_CLIENT_ID` and `FUNCX_SDK_CLIENT_SECRET` environment variables are set and match the ones the deployment will use (so that those credentials will "own" the endpoint uuid).

### Install the helm chart

Create a local values.yaml file (see `garden_values.yaml`) to set any specific values you wish to
override. Then invoke the chart installation by specifying your custom values and this globus-compute-endpoint chart:

```shell script
helm install -n dlhub -f ./garden_values.yaml garden-demo ./globus_compute_endpoint
```

---

And view the pod's status via:

```shell script
kubectl get pods [endpoint pod name]
```

Or its logs via:

```shell script
kubectl logs [endpoint pod name]
```

## Values
The deployment is configured via values.yaml file. See `./globus_compute_endpoint/values.yaml` for defaults and `./garden_values.yaml` to see what we override.

## Globus Group
Since the endpoint can't be public, we've set it up behind a "Garden Users" Group that people can join. To associate an endpoint uuid with a group uuid you need to email support@globus.org and someone on the compute team will handle it.

### Changes to the chart

I cross-referenced a few related versions of this chart that I was able to find to put this one together -- the first is the one on github, and the other two are from the funcx helm repo (`helm repo add funcx http://funcx.org/funcx-helm-charts`). The main difference between the `funcx/globus-compute-endpoint` chart pulled via helm and the one pulled from github is that the `templates/endpoint-instance-config.yaml`file templates a `config.py`in the former and a `config.yaml` in the latter. My understanding is that good globus-compute-practices is to prefer the `config.yaml`-style (as seen  [here](https://github.com/funcx-faas/funcX/blob/4fc7047648693fbb688c93f8a09b4aeb830b10bd/helm/funcx_endpoint/templates/endpoint-instance-config.yaml#L11) on the github one), but I wasn't able to get that to work.

So I stuck with a `config.py`-style template from the http://funcx.org/funcx-helm-charts one, with a slight change to use     `HighThroughputEngine` instead of the default `HighThroughputExecutor`, which has apparently been deprecated.

I also tweaked a couple of broken defaults in the chart's `values.yaml`, namely `funcXServiceAddress` from `curl https://compute.api.globus.org` to `https://compute.api.globus.org` (the `curl` was causing some weird network errors) and `image.repository` from `compute/kube-endpoint` back to `funcx/kube-endpoint` (there is no `compute/kube-endpoint` repo on dockerhub).

TLDR: outside of `./globus_compute_endpoint/values.yaml` and `./globus_compute_endpoint/templates/endpoint-instance-config.yaml` the chart should be identical to the one on github.