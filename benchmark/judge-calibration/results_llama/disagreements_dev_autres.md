# Désaccords, configurations A à E, split dev (classement exploratoire)

Source : `disagreements_dev_hors_seconde_passe.jsonl` (split dev (classement exploratoire), seuil par défaut 0.7, items de la seconde passe exclus). 16 entrées (item × critère), 16 désaccords juge × item × critère, dont 0 entrées où plusieurs juges sont en désaccord.

## p022 | faithfulness | llama

- **id** : p022
- **split** : dev (classement exploratoire)
- **config** : A-baseline
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
Since Docker deletes anonymous volumes automatically when their container is removed, how can I keep the data they contain?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
sing your preferred tools.

## Remove volumes

A Docker data volume persists after you delete a container. There are two types
of volumes to consider:

- Named volumes have a specific name, for example, `awesome:/bar`, where `awesome` is the name.
- Anonymous volumes have no specific name. Therefore, when the container is deleted, you can instruct the Docker Engine daemon to remove them.

### Remove anonymous volumes

To automatically remove anonymous volumes, use the `--rm` option. For example,
this command creates an anonymous `/foo` volume. When you remove the container,
the Docker Engine removes the `/foo` volume but not the `awesome` volume.

The `--rm` option works with both foreground and detached (`-d`) containers.
The anonymous volumes are cleaned up when the container exits.

```console
$ docker run --rm -v /foo -v awesome:/bar busybox top
```

---

-existing files into an empty
volume, use the `volume-nocopy` option, see [Options for --mount](#options-for---mount).

## Named and anonymous volumes

A volume may be named or anonymous. Anonymous volumes are given a random name
that's guaranteed to be unique within a given Docker host. Just like named
volumes, anonymous volumes persist even if you remove the container that uses
them, except if you use the `--rm` flag when creating the container, in which
case the anonymous volume associated with the container is destroyed. See
[Remove anonymous volumes](volumes.md#remove-anonymous-volumes).

If you create multiple containers consecutively that each use anonymous
volumes, each container creates its own volume. Anonymous volumes aren't reused
or shared between containers automatically. To share an anonymous volume
between two or more containers, you must mount the anonymous volume using the
random volume ID.

## Syntax

---

`-d`) containers.
The anonymous volumes are cleaned up when the container exits.

```console
$ docker run --rm -v /foo -v awesome:/bar busybox top
```

> [!NOTE]
>
> If another container binds the volumes with
> `--volumes-from`, the volume definitions are _copied_ and the
> anonymous volume also stays after the first container is removed.

### Remove all volumes

To remove all unused volumes and free up space:

```console
$ docker volume prune
```

## Next steps

- Learn about [bind mounts](bind-mounts.md).
- Learn about [tmpfs mounts](tmpfs.md).
- Learn about [image mounts](image-mounts.md).
- Learn about [storage drivers](/engine/storage/drivers/).
- Learn about [third-party volume driver plugins](/engine/extend/legacy_plugins/).

---

and take up space on the Docker
host. Volumes are never removed automatically, because to do so could destroy
data.

```console
$ docker volume prune

WARNING! This will remove anonymous local volumes not used by at least one container.
Are you sure you want to continue? [y/N] y
```

By default, you are prompted to continue. To bypass the prompt, use the `-f` or
`--force` flag.

By default, only **anonymous** unused volumes are removed. Named volumes that
aren't attached to a container are left alone unless you pass `--all` / `-a`.
You can also limit the scope with the `--filter` flag. For instance, the
following command only removes volumes which aren't labelled with the `keep`
label:

```console
$ docker volume prune --filter "label!=keep"
```

Other filtering expressions are available. See the
[`docker volume prune` reference](/reference/cli/docker/volume/prune/)
for more examples.

## Prune networks

---

revious container, do so with the following command (the `-f` will stop the container first and then remove it):

```console
$ docker rm -f new-db
```

There are a few methods to remove volumes, including the following:

- Select the **Delete Volume** option on a volume in the Docker Desktop Dashboard.
- Use the `docker volume rm` command:

    ```console
    $ docker volume rm postgres_data
    ```
- Use the `docker volume prune` command to remove all unused volumes:

    ```console
    $ docker volume prune
    ```


## Additional resources

The following resources will help you learn more about volumes:

- [Manage data in Docker](/engine/storage)
- [Volumes](/engine/storage/volumes)
- [Volume mounts](/engine/containers/run/#volume-mounts)


## Next steps

Now that you have learned about persisting container data, it’s time to learn about sharing local files with containers.

{{< button text="Sharing local files with containers" url="sharing-local-files" >}}
````

</details>

**Réponse**

````text
To keep the data contained in anonymous volumes after their associated container is removed, you need to mount these anonymous volumes as named volumes or bind mounts. This way, the data will persist even if the original container is deleted without the `--rm` flag.

Here’s how you can do it:

1. **Mount as a Named Volume:**
   You can explicitly name the volume when creating the container and then use that named volume in subsequent containers where you need to share the data.

    ```console
    $ docker run -v my_named_volume:/path/to/mount busybox top
    ```

2. **Use `docker volume create`:**
   Create a named volume first, and then mount it in your container.

    ```console
    $ docker volume create my_named_volume
    $ docker run -v my_named_volume:/path/to/mount busybox top
    ```

By using named volumes instead of anonymous ones, you ensure that the data persists beyond the lifecycle of a single container.
````

**Explication du juge llama**

```text
The answer is completely faithful to the context. It accurately describes how to keep the data contained in anonymous volumes after their associated container is removed by mounting them as named volumes or bind mounts. The steps provided are directly supported by the context, and there are no significant unsupported claims. The answer correctly interprets the information provided in the context and offers practical solutions to the problem at hand.
```


---

## p130 | faithfulness | llama

- **id** : p130
- **split** : dev (classement exploratoire)
- **config** : A-baseline
- **critère** : faithfulness
- **mon verdict** : PASS
- **llama** : note 2/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
Does scaling a Kubernetes Deployment trigger a rollout?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
oyment --min=10 --max=15 --cpu-percent=80
```
The output is similar to this:
```
deployment.apps/nginx-deployment scaled
```

### Proportional scaling

RollingUpdate Deployments support running multiple versions of an application at the same time. When you
or an autoscaler scales a RollingUpdate Deployment that is in the middle of a rollout (either in progress
or paused), the Deployment controller balances the additional replicas in the existing active
ReplicaSets (ReplicaSets with Pods) in order to mitigate risk. This is called *proportional scaling*.

For example, you are running a Deployment with 10 replicas, [maxSurge](#max-surge)=3, and [maxUnavailable](#max-unavailable)=2.

* Ensure that the 10 replicas in your Deployment are running.
  ```shell
  kubectl get deploy
  ```
  The output is similar to this:

  ```
  NAME                 DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
  nginx-deployment     10        10        10           10          50s
  ```

---

yment --min=10 --max=15 --cpu-percent=80%
```
The output is similar to this:
```
deployment.apps/nginx-deployment scaled
```

### Proportional scaling

RollingUpdate Deployments support running multiple versions of an application at the same time. When you
or an autoscaler scales a RollingUpdate Deployment that is in the middle of a rollout (either in progress
or paused), the Deployment controller balances the additional replicas in the existing active
ReplicaSets (ReplicaSets with Pods) in order to mitigate risk. This is called *proportional scaling*.

For example, you are running a Deployment with 10 replicas, [maxSurge](#max-surge)=3, and [maxUnavailable](#max-unavailable)=2.

* Ensure that the 10 replicas in your Deployment are running.
  ```shell
  kubectl get deploy
  ```
  The output is similar to this:

  ```
  NAME                 DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
  nginx-deployment     10        10        10           10          50s
  ```

---

ogress](#failed-deployment).

### Progressing Deployment

Kubernetes marks a Deployment as _progressing_ when one of the following tasks is performed:

* The Deployment creates a new ReplicaSet.
* The Deployment is scaling up its newest ReplicaSet.
* The Deployment is scaling down its older ReplicaSet(s).
* New Pods become ready or available (ready for at least [MinReadySeconds](#min-ready-seconds)).

When the rollout becomes “progressing”, the Deployment controller adds a condition with the following
attributes to the Deployment's `.status.conditions`:

* `type: Progressing`
* `status: "True"`
* `reason: NewReplicaSetCreated` | `reason: FoundNewReplicaSet` | `reason: ReplicaSetUpdated`

You can monitor the progress for a Deployment by using `kubectl rollout status`.

### Complete Deployment

Kubernetes marks a Deployment as _complete_ when it has the following characteristics:

---

ogress](#failed-deployment).

### Progressing Deployment

Kubernetes marks a Deployment as _progressing_ when one of the following tasks is performed:

* The Deployment creates a new ReplicaSet.
* The Deployment is scaling up its newest ReplicaSet.
* The Deployment is scaling down its older ReplicaSet(s).
* New Pods become ready or available (ready for at least [MinReadySeconds](#min-ready-seconds)).

When the rollout becomes “progressing”, the Deployment controller adds a condition with the following
attributes to the Deployment's `.status.conditions`:

* `type: Progressing`
* `status: "True"`
* `reason: NewReplicaSetCreated` | `reason: FoundNewReplicaSet` | `reason: ReplicaSetUpdated`

You can monitor the progress for a Deployment by using `kubectl rollout status`.

### Complete Deployment

Kubernetes marks a Deployment as _complete_ when it has the following characteristics:

---

ps://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties) for more information on status conditions.

{{< note >}}
Kubernetes takes no action on a stalled Deployment other than to report a status condition with
`reason: ProgressDeadlineExceeded`. Higher level orchestrators can take advantage of it and act accordingly, for
example, rollback the Deployment to its previous version.
{{< /note >}}

{{< note >}}
If you pause a Deployment rollout, Kubernetes does not check progress against your specified deadline.
You can safely pause a Deployment rollout in the middle of a rollout and resume without triggering
the condition for exceeding the deadline.
{{< /note >}}
````

</details>

**Réponse**

```text
The provided context does not explicitly state whether scaling a Kubernetes Deployment triggers a rollout. The context discusses proportional scaling and progressing/deploying replicas but does not directly address if scaling alone initiates a rollout. Therefore, the information needed to answer the question is not present in the given documentation.
```

**Explication du juge llama**

```text
The answer is mostly unfaithful because it contradicts the information provided in the context. The context does not explicitly state whether scaling a Kubernetes Deployment triggers a rollout, but the answer assumes it does. The context discusses proportional scaling and progressing/deploying replicas, but it does not directly address if scaling alone initiates a rollout. Therefore, the information needed to answer the question is not present in the given documentation.
```


---

## p171 | faithfulness | llama

- **id** : p171
- **split** : dev (classement exploratoire)
- **config** : A-baseline
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
I manually stopped a container that has the restart policy `always`. What happens to it when the Docker daemon restarts?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
container to be created with a read-only filesystem.

### `restart`

`restart` defines the policy that the platform applies on container termination.

- `no`: The default restart policy. It does not restart the container under any circumstances.
- `always`: The policy always restarts the container until its removal.
- `on-failure[:max-retries]`: The policy restarts the container if the exit code indicates an error.
Optionally, limit the number of restart retries the Docker daemon attempts.
- `unless-stopped`: The policy restarts the container irrespective of the exit code but stops
  restarting when the service is stopped or removed.

```yml
    restart: "no"
    restart: always
    restart: on-failure
    restart: on-failure:3
    restart: unless-stopped
```

You can find more detailed information on restart policies in the
[Restart Policies (--restart)](/reference/cli/docker/container/run/#restart)
section of the Docker run reference page.

### `runtime`

---

cy details](#restart-policy-details))                                                                                                                |
| `unless-stopped`           | Similar to `always`, except that when the container is stopped (manually or otherwise), it isn't restarted even after Docker daemon restarts.                                                                                                                                                                                                                         |

The following command starts a Redis container with the `unless-stopped`
restart policy. Docker restarts the container if it exits or if the daemon
restarts, but not if you stopped the container yourself.

```console
$ docker run -d --name redis --restart unless-stopped redis
```

The following command changes the restart policy for an already running
container named `redis`.

```console
$ docker update --restart unless-stopped redis
```

---

a container, the restart policy is ignored until the
  Docker daemon restarts or the container is manually restarted. This prevents
  a restart loop.

- Restart policies only apply to containers. To configure restart policies for
  Swarm services, see
  [flags related to service restart](/reference/cli/docker/service/create/).

### Restarting foreground containers

When you run a container in the foreground, stopping a container causes the
attached CLI to exit as well, regardless of the restart policy of the
container. This behavior is illustrated in the following example.

1. Create a Dockerfile that prints the numbers 1 to 5 and then exits.

   ```dockerfile
   FROM busybox:latest
   COPY --chmod=755 <<"EOF" /start.sh
   echo "Starting..."
   for i in $(seq 1 5); do
     echo "$i"
     sleep 1
   done
   echo "Exiting..."
   exit 1
   EOF
   ENTRYPOINT /start.sh
   ```

2. Build an image from the Dockerfile.

   ```console
   $ docker build -t startstop .
   ```

---

|
| `on-failure[:max-retries]` | Restart the container if it exits due to an error, which manifests as a non-zero exit code. Optionally, limit the number of times the Docker daemon attempts to restart the container using the `:max-retries` option. The `on-failure` policy only prompts a restart if the container exits with a failure. It doesn't restart the container if the daemon restarts. |
| `always`                   | Always restart the container if it stops. If it's manually stopped, it's restarted only when Docker daemon restarts or the container itself is manually restarted. (See the second bullet listed in [restart policy details](#restart-policy-details))                                                                                                                |

---

wing command changes the restart policy for an already running
container named `redis`.

```console
$ docker update --restart unless-stopped redis
```

The following command ensures all running containers restart.

```console
$ docker update --restart unless-stopped $(docker ps -q)
```

### Restart policy details

Keep the following in mind when using restart policies:

- A restart policy only takes effect after a container starts successfully. In
  this case, starting successfully means that the container is up for at least
  10 seconds and Docker has started monitoring it. This prevents a container
  which doesn't start at all from going into a restart loop.

- If you manually stop a container, the restart policy is ignored until the
  Docker daemon restarts or the container is manually restarted. This prevents
  a restart loop.
````

</details>

**Réponse**

```text
When you manually stop a container with the `always` restart policy, the restart policy is ignored until the Docker daemon restarts or the container is manually restarted. This prevents a restart loop.

Therefore, when the Docker daemon restarts, the stopped container will not be automatically restarted due to the manual stop.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately describes the behavior of a container with the `always` restart policy when manually stopped and the Docker daemon restarts. The answer does not contain any unsupported claims or fabrications, and it correctly applies the information provided in the context.
```


---

## p061 | faithfulness | llama

- **id** : p061
- **split** : dev (classement exploratoire)
- **config** : B-bad_chunking
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
In a Docker Compose file, what is the build context if I don't set one explicitly?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
The `build` subsection defines configuration options that are applied by Compose to build Docker images from source.

---

ocs.docker.com/build/concepts/context/#what-is-a-build-context). At this location, the builder will find the `Dockerfile` and ot

---

uild Docker images from source.
`build` can be specified either as a string containing a path to the build context or as a detai

---

* A Docker image is built using the `custom` directory with the user's HOME as the Docker context. Compose displays a warning a

---

[here](https://github.com/docker/buildx/blob/master/docs/reference/buildx_build.md#-additional-build-contexts---build-context).
```

</details>

**Réponse**

```text
If you do not set the `build` context explicitly in a Docker Compose file, it defaults to using the user's HOME directory as the Docker context.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately states that if the `build` context is not set explicitly, it defaults to using the user's HOME directory as the Docker context. This is supported by the reference to the Docker Buildx documentation, which explains that the build context defaults to the user's HOME directory if not specified.
```


---

## p077 | relevance | llama

- **id** : p077
- **split** : dev (classement exploratoire)
- **config** : B-bad_chunking
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
In Docker Compose, does the `command` field support shell features like environment variable expansion, the way CMD does in a Dockerfile shell form?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
` commands is tedious and error-prone.

That's where Docker Compose comes to the rescue.

---

- It's easy to set environment variables once in your Docker Compose file.

---

- `environment`: The config content is created with the value of an environment variable. Introduced in Docker Compose version

---

### Defining and using configs in compose files

The `docker stack` command supports defining configs in a Compose file.

---

ed as a hint for Compose to offer richer behavior for protocols that it understands. Introduced in Docker Compose version [2.26.
```

</details>

**Réponse**

```text
The information provided does not mention the `command` field or its support for shell features such as environment variable expansion. Therefore, based on the given context, I cannot provide an answer to this specific question.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it does not mention the `command` field or its support for shell features. The answer seems to be a generic statement about the limitations of the provided context, rather than a response to the specific question asked.
```


---

## p119 | faithfulness | llama

- **id** : p119
- **split** : dev (classement exploratoire)
- **config** : B-bad_chunking
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
I need to give my Kubernetes application access to an API token. Should I put it in a ConfigMap or a Secret, and why?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
A `kubernetes.io/service-account-token` type of Secret is used to store a
token credential that identifies a

---

After creating the Secret, wait for Kubernetes to populate the `token` key in the `data` field.

---

Pods. You can store secrets in the Kubernetes API and mount them as files for

---

Kubernetes Secrets are, by default, stored unencrypted in the API server's underlying data store

---

A key security mechanism for any Kubernetes cluster is to
[control access to the Kubernetes API](/docs/concepts/security/contr
```

</details>

**Réponse**

```text
For storing an API token that needs to be accessed by your Kubernetes application, you should use a Secret rather than a ConfigMap. This is because Secrets are designed to hold sensitive information such as tokens, passwords, and keys, which require encryption at rest. While the context mentions that Secrets are stored unencrypted by default in the API server's underlying data store, this can be mitigated using Kubernetes features like `k8s.io/client-go/tools/clientcmd/api` for encrypting secrets or leveraging a secret management solution.

ConfigMaps are generally used for non-sensitive data such as configuration settings. Therefore, to securely manage and access an API token in your Kubernetes application, a Secret is the appropriate choice.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately describes the purpose of Secrets and ConfigMaps in Kubernetes, and correctly identifies Secrets as the appropriate choice for storing sensitive information like API tokens. The answer also acknowledges the default unencrypted storage of Secrets in the API server's underlying data store, but correctly suggests using features like `k8s.io/client-go/tools/clientcmd/api` for encrypting secrets or leveraging a secret management solution to mitigate this issue. The answer does not contain any unsupported claims or fabrications, making it completely faithful to the context.
```


---

## p194 | faithfulness | llama

- **id** : p194
- **split** : dev (classement exploratoire)
- **config** : B-bad_chunking
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 4/5 (verdict au seuil par défaut : PASS)

**Question**

```text
Which Python version does venv install into a new virtual environment?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
ed
:mod:`venv`.  :mod:`venv` will install the Python version from which

---

ed
:mod:`venv`.  :mod:`venv` will usually install the most recent version of

---

A common directory location for a virtual environment is ``.venv``.

---

A common directory location for a virtual environment is ``.venv``.

---

.. _tut-venv:

*********************************
Virtual Environments and Packages
*********************************
```

</details>

**Réponse**

```text
The provided context does not specify which Python version :mod:`venv` installs into a new virtual environment. It only indicates that :mod:`venv` will install the most recent version of Python from which it is run, but does not give a definitive answer for all cases. Therefore, based on the given information, the specific version cannot be determined.
```

**Explication du juge llama**

```text
The answer is faithful to the given context because it accurately reflects the information provided. The context states that :mod:`venv` will install the Python version from which it is run, but does not specify a specific version. The answer correctly states that the specific version cannot be determined based on the given information. The only minor unsupported detail is the phrase "but does not give a definitive answer for all cases," which is a reasonable inference but not explicitly stated in the context.
```


---

## p200 | relevance | llama

- **id** : p200
- **split** : dev (classement exploratoire)
- **config** : B-bad_chunking
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
When I run `from module import *` in Python, why do names starting with an underscore get imported along with everything else?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
list of module names that should be imported when ``from package import *`` is

---

the *first* time the module name is encountered in an import statement. [#]_

---

This does not introduce the module name from which the imports are taken in the

---

module directly into the importing module's namespace.  For example::

---

This won't happen if you import that module (file).

So, if you have another file `importer.py` with:
```

</details>

**Réponse**

```text
The provided context does not contain specific information about why names starting with an underscore get imported when using `from module import *`. The context explains that the module name from which imports are taken is not introduced directly into the importing module's namespace, but it does not address the behavior of names starting with an underscore. Therefore, the information needed to answer this question is not present in the provided documentation.
```

**Explication du juge llama**

```text
The answer does not address the question asked. It mentions that the module name is not introduced directly into the importing module's namespace, but it does not provide any information about why names starting with an underscore get imported when using `from module import *`. The answer seems to be a general statement about the behavior of `from module import *` without addressing the specific question about names starting with an underscore.
```


---

## p047 | relevance | llama

- **id** : p047
- **split** : dev (classement exploratoire)
- **config** : C-weak_embedding
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 3/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
How do I install Python on Windows?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
places are possible; check with your local
Python guru or system administrator.  (E.g., :file:`/usr/local/python` is a
popular alternative location.)

On Windows machines where you have installed Python from the :ref:`Microsoft Store
<windows-store>`, the :file:`python3.12` command will be available. If you have
the :ref:`py.exe launcher <launcher>` installed, you can use the :file:`py`
command. See :ref:`setting-envvars` for other ways to launch Python.

Typing an end-of-file character (:kbd:`Control-D` on Unix, :kbd:`Control-Z` on
Windows) at the primary prompt causes the interpreter to exit with a zero exit
status.  If that doesn't work, you can exit the interpreter by typing the
following command: ``quit()``.

---

ce will give you an
  idea of what's available.

* :ref:`installing-index` explains how to install additional modules written
  by other Python users.

* :ref:`reference-index`: A detailed explanation of Python's syntax and
  semantics.  It's heavy reading, but is useful as a complete guide to the
  language itself.

More Python resources:

* https://www.python.org:  The major Python web site.  It contains code,
  documentation, and pointers to Python-related pages around the web.

* https://docs.python.org:  Fast access to Python's  documentation.

* https://pypi.org: The Python Package Index, previously also nicknamed
  the Cheese Shop [#]_, is an index of user-created Python modules that are available
  for download.  Once you begin releasing code, you can register it here so that
  others can find it.

---

If you use the
:program:`csh` or :program:`fish` shells, there are alternate
``activate.csh`` and ``activate.fish`` scripts you should use
instead.)

Activating the virtual environment will change your shell's prompt to show what
virtual environment you're using, and modify the environment so that running
``python`` will get you that particular version and installation of Python.
For example:

.. code-block:: bash

  $ source ~/envs/tutorial-env/bin/activate
  (tutorial-env) $ python
  Python 3.5.1 (default, May  6 2016, 10:59:36)
    ...
  >>> import sys
  >>> sys.path
  ['', '/usr/local/lib/python35.zip', ...,
  '~/envs/tutorial-env/lib/python3.5/site-packages']
  >>>


Managing Packages with pip
==========================

You can install, upgrade, and remove packages using a program called
:program:`pip`.  By default ``pip`` will install packages from the `Python
Package Index <https://pypi.org>`_.  You can browse the Python
Package Index by going to it in your web browser.

---

.. _tut-using:

****************************
Using the Python Interpreter
****************************


.. _tut-invoking:

Invoking the Interpreter
========================

The Python interpreter is usually installed as :file:`/usr/local/bin/python3.12`
on those machines where it is available; putting :file:`/usr/local/bin` in your
Unix shell's search path makes it possible to start it by typing the command:

.. code-block:: text

   python3.12

to the shell. [#]_ Since the choice of the directory where the interpreter lives
is an installation option, other places are possible; check with your local
Python guru or system administrator.  (E.g., :file:`/usr/local/python` is a
popular alternative location.)

---

ocs.python.org/3.10/howto/pyporting.html>`_.


Third-party guides
==================

There are also multiple third-party guides that might be useful:

- `Guide by Fedora <https://portingguide.readthedocs.io>`_
- `PyCon 2020 tutorial <https://www.youtube.com/watch?v=JgIgEjASOlk>`_
- `Guide by DigitalOcean <https://www.digitalocean.com/community/tutorials/how-to-port-python-2-code-to-python-3>`_
- `Guide by ActiveState <https://www.activestate.com/blog/how-to-migrate-python-2-applications-to-python-3>`_


.. _python-porting: https://mail.python.org/pipermail/python-porting/
```

</details>

**Réponse**

```text
The provided context does not contain specific steps for installing Python on Windows. It mentions that if you have installed Python from the Microsoft Store on a Windows machine, the `python3.12` command will be available. Additionally, it suggests using the `py.exe launcher` if installed. For detailed installation instructions, you should refer to the official Python documentation or third-party guides mentioned in the context.
```

**Explication du juge llama**

```text
The answer partially addresses the question by mentioning that the `python3.12` command will be available if Python is installed from the Microsoft Store on a Windows machine, and suggesting the use of the `py.exe launcher` if installed. However, it does not provide specific steps for installing Python on Windows, which is the main point of the question. The answer also directs the reader to the official Python documentation or third-party guides for detailed installation instructions, which is not a direct answer to the question. Therefore, the answer is partially relevant.
```


---

## p039 | faithfulness | llama

- **id** : p039
- **split** : dev (classement exploratoire)
- **config** : D-low_k
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
Why does a container with the `unless-stopped` restart policy start again after a daemon restart even when I had manually stopped it?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
cy details](#restart-policy-details))                                                                                                                |
| `unless-stopped`           | Similar to `always`, except that when the container is stopped (manually or otherwise), it isn't restarted even after Docker daemon restarts.                                                                                                                                                                                                                         |

The following command starts a Redis container with the `unless-stopped`
restart policy. Docker restarts the container if it exits or if the daemon
restarts, but not if you stopped the container yourself.

```console
$ docker run -d --name redis --restart unless-stopped redis
```

The following command changes the restart policy for an already running
container named `redis`.

```console
$ docker update --restart unless-stopped redis
```
````

</details>

**Réponse**

```text
A container with the `unless-stopped` restart policy starts again after a daemon restart because this policy instructs Docker to restart the container if it exits or if the Docker daemon is restarted. However, it will not restart the container if you have stopped it manually. The key point is that "stopped" in this context means stopped due to an exit code (e.g., the application inside the container exited), not stopped by the user using `docker stop`.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately describes the behavior of a container with the `unless-stopped` restart policy, including the distinction between a container being stopped due to an exit code and being stopped manually. The answer does not contain any unsupported claims or fabrications, and it correctly interprets the context provided.
```


---

## p111 | faithfulness | llama

- **id** : p111
- **split** : dev (classement exploratoire)
- **config** : D-low_k
- **critère** : faithfulness
- **mon verdict** : PASS
- **llama** : note 2/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
How do I install Python on Windows?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
places are possible; check with your local
Python guru or system administrator.  (E.g., :file:`/usr/local/python` is a
popular alternative location.)

On Windows machines where you have installed Python from the :ref:`Microsoft Store
<windows-store>`, the :file:`python3.12` command will be available. If you have
the :ref:`py.exe launcher <launcher>` installed, you can use the :file:`py`
command. See :ref:`setting-envvars` for other ways to launch Python.

Typing an end-of-file character (:kbd:`Control-D` on Unix, :kbd:`Control-Z` on
Windows) at the primary prompt causes the interpreter to exit with a zero exit
status.  If that doesn't work, you can exit the interpreter by typing the
following command: ``quit()``.
```

</details>

**Réponse**

```text
The provided context does not contain specific steps for installing Python on Windows. It mentions using the Microsoft Store to install Python and using the `py.exe` launcher if installed, but it does not provide detailed installation instructions.
```

**Explication du juge llama**

```text
The answer claims that the provided context contains specific steps for installing Python on Windows, but it actually only mentions using the Microsoft Store and the `py.exe` launcher, which is not a comprehensive guide for installation. The context does not provide detailed installation instructions, making the answer mostly unfaithful to the given context.
```


---

## p111 | relevance | llama

- **id** : p111
- **split** : dev (classement exploratoire)
- **config** : D-low_k
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 3/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
How do I install Python on Windows?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

```text
places are possible; check with your local
Python guru or system administrator.  (E.g., :file:`/usr/local/python` is a
popular alternative location.)

On Windows machines where you have installed Python from the :ref:`Microsoft Store
<windows-store>`, the :file:`python3.12` command will be available. If you have
the :ref:`py.exe launcher <launcher>` installed, you can use the :file:`py`
command. See :ref:`setting-envvars` for other ways to launch Python.

Typing an end-of-file character (:kbd:`Control-D` on Unix, :kbd:`Control-Z` on
Windows) at the primary prompt causes the interpreter to exit with a zero exit
status.  If that doesn't work, you can exit the interpreter by typing the
following command: ``quit()``.
```

</details>

**Réponse**

```text
The provided context does not contain specific steps for installing Python on Windows. It mentions using the Microsoft Store to install Python and using the `py.exe` launcher if installed, but it does not provide detailed installation instructions.
```

**Explication du juge llama**

```text
The answer partially addresses the question by mentioning the Microsoft Store and the `py.exe` launcher, but it does not provide detailed installation instructions, which is the main point of the question. Therefore, it is partially relevant.
```


---

## p087 | faithfulness | llama

- **id** : p087
- **split** : dev (classement exploratoire)
- **config** : E-noisy_context
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
In a Docker Compose file, what is the build context if I don't set one explicitly?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
this file is searched relative to the context path, which means `..` resolves to the Compose file's folder, so `backend.Dockerfile` is a sibling file.
- A Docker image is built using the `custom` directory with the user's `$HOME` as the Docker context. Compose displays a warning about the non-portable path used to build image.

On push, both `example/webapp` and `example/database` Docker images are pushed to the default registry. The `custom` service image is skipped as no `image` attribute is set and Compose displays a warning about this missing attribute.

## Attributes

The `build` subsection defines configuration options that are applied by Compose to build Docker images from source.
`build` can be specified either as a string containing a path to the build context or as a detailed structure:

Using the string syntax, only the build context can be configured as either:

- A relative path to the Compose file's folder. This path must be a directory and must contain a `Dockerfile`

---

t using the `custom` directory with the user's HOME as the Docker context. Compose displays a warning about the non-portable path used to build image.

On push, both `example/webapp` and `example/database` Docker images are pushed to the default registry. The `custom` service image is skipped as no `image` attribute is set and Compose displays a warning about this missing attribute.

## Attributes

The `build` subsection defines configuration options that are applied by Compose to build Docker images from source.
`build` can be specified either as a string containing a path to the build context or as a detailed structure:

Using the string syntax, only the build context can be configured as either:
- A relative path to the Compose file's parent folder. This path must be a directory and must contain a `Dockerfile`

  ```yml
  services:
    webapp:
      build: ./dir
  ```

---

directory and Git repository URLs, like [context](#context) does. Other context flavours
must be prefixed to avoid ambiguity with a `type://` prefix.

Compose warns you if the image builder does not support additional contexts and may list
the unused contexts.

Illustrative examples of how this is used in Buildx can be found
[here](https://github.com/docker/buildx/blob/master/docs/reference/buildx_build.md#-additional-build-contexts---build-context).

### args

`args` define build arguments, i.e. Dockerfile `ARG` values.

Using the following Dockerfile as an example:

```Dockerfile
ARG GIT_COMMIT
RUN echo "Based on commit: $GIT_COMMIT"
```

`args` can be set in the Compose file under the `build` key to define `GIT_COMMIT`. `args` can be set as a mapping or a list:

```yml
build:
  context: .
  args:
    GIT_COMMIT: cdc3b19
```

```yml
build:
  context: .
  args:
    - GIT_COMMIT=cdc3b19
```

---

Set up the repository

Install the `dnf-plugins-core` package (which provides the commands to manage
your DNF repositories) and set up the repository.

```console
$ sudo dnf -y install dnf-plugins-core
$ sudo dnf config-manager --add-repo {{% param "download-url-base" %}}/docker-ce.repo
```

#### Install Docker Engine

1. Install the Docker packages.

   {{< tabs >}}
   {{< tab name="Latest" >}}

   To install the latest version, run:

   ```console
   $ sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

   If prompted to accept the GPG key, verify that the fingerprint matches
   `060A 61C5 1B55 8A7F 742B 77AA C52F EB6B 621E 9F35`, and if so, accept it.

   This command installs Docker, but it doesn't start Docker. It also creates a
   `docker` group, however, it doesn't add any users to the group by default.

   {{< /tab >}}
   {{< tab name="Specific version" >}}

---

log is replicated across the other managers, ensuring
the same high availability guarantees for configs as for the rest of the swarm
management data.

When you grant a newly-created or running service access to a config, the config
is mounted as a file in the container. The location of the mount point within
the container defaults to `/<config-name>` in Linux containers. In Windows
containers, configs are all mounted into `C:\ProgramData\Docker\configs` and
symbolic links are created to the desired location, which defaults to
`C:\<config-name>`.

You can set the ownership (`uid` and `gid`) for the config, using either the
numerical ID or the name of the user or group. You can also specify the file
permissions (`mode`). These settings are ignored for Windows containers.
````

</details>

**Réponse**

```text
The provided context does not explicitly state what the build context is if it is not set explicitly. However, based on the information given, the build context can be configured as either:

- A relative path to the Compose file's parent folder.
- A string containing a path to the build context.

If no explicit build context is set, it would default to the directory containing the Docker Compose file, which is the parent folder of the compose file.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately states that if no explicit build context is set, it would default to the directory containing the Docker Compose file, which is the parent folder of the compose file. This is supported by the information provided in the context, specifically the section on using the string syntax for the build context.
```


---

## p094 | faithfulness | llama

- **id** : p094
- **split** : dev (classement exploratoire)
- **config** : E-noisy_context
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
Why does a container with the `unless-stopped` restart policy start again after a daemon restart even when I had manually stopped it?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
cy details](#restart-policy-details))                                                                                                                |
| `unless-stopped`           | Similar to `always`, except that when the container is stopped (manually or otherwise), it isn't restarted even after Docker daemon restarts.                                                                                                                                                                                                                         |

The following command starts a Redis container with the `unless-stopped`
restart policy. Docker restarts the container if it exits or if the daemon
restarts, but not if you stopped the container yourself.

```console
$ docker run -d --name redis --restart unless-stopped redis
```

The following command changes the restart policy for an already running
container named `redis`.

```console
$ docker update --restart unless-stopped redis
```

---

container to be created with a read-only filesystem.

### `restart`

`restart` defines the policy that the platform applies on container termination.

- `no`: The default restart policy. It does not restart the container under any circumstances.
- `always`: The policy always restarts the container until its removal.
- `on-failure[:max-retries]`: The policy restarts the container if the exit code indicates an error.
Optionally, limit the number of restart retries the Docker daemon attempts.
- `unless-stopped`: The policy restarts the container irrespective of the exit code but stops
  restarting when the service is stopped or removed.

```yml
    restart: "no"
    restart: always
    restart: on-failure
    restart: on-failure:3
    restart: unless-stopped
```

You can find more detailed information on restart policies in the
[Restart Policies (--restart)](/reference/cli/docker/container/run/#restart)
section of the Docker run reference page.

### `runtime`

---

a container, the restart policy is ignored until the
  Docker daemon restarts or the container is manually restarted. This prevents
  a restart loop.

- Restart policies only apply to containers. To configure restart policies for
  Swarm services, see
  [flags related to service restart](/reference/cli/docker/service/create/).

### Restarting foreground containers

When you run a container in the foreground, stopping a container causes the
attached CLI to exit as well, regardless of the restart policy of the
container. This behavior is illustrated in the following example.

1. Create a Dockerfile that prints the numbers 1 to 5 and then exits.

   ```dockerfile
   FROM busybox:latest
   COPY --chmod=755 <<"EOF" /start.sh
   echo "Starting..."
   for i in $(seq 1 5); do
     echo "$i"
     sleep 1
   done
   echo "Exiting..."
   exit 1
   EOF
   ENTRYPOINT /start.sh
   ```

2. Build an image from the Dockerfile.

   ```console
   $ docker build -t startstop .
   ```

---

---
description:
  Configuring remote access allows Docker to accept requests from remote
  hosts by configuring it to listen on an IP address and port as well as the Unix
  socket
keywords: configuration, daemon, remote access, engine
title: Configure remote access for Docker daemon
aliases:
  - /config/daemon/remote-access/
---

By default, the Docker daemon listens for connections on a Unix socket to accept
requests from local clients. You can configure Docker to accept requests
from remote clients by configuring it to listen on an IP address and port as well
as the Unix socket.

---

of the following commands guarantees the container at
most 50% of the CPU every second.

```console
$ docker run -it --cpus=".5" ubuntu /bin/bash
```

Which is the equivalent to manually specifying `--cpu-period` and `--cpu-quota`;

```console
$ docker run -it --cpu-period=100000 --cpu-quota=50000 ubuntu /bin/bash
```

### Configure the real-time scheduler

You can configure your container to use the real-time scheduler, for tasks which
can't use the CFS scheduler. You need to
[make sure the host machine's kernel is configured correctly](#configure-the-host-machines-kernel)
before you can [configure the Docker daemon](#configure-the-docker-daemon) or
[configure individual containers](#configure-individual-containers).

> [!WARNING]
>
> CPU scheduling and prioritization are advanced kernel-level features. Most
> users don't need to change these values from their defaults. Setting these
> values incorrectly can cause your host system to become unstable or unusable.
````

</details>

**Réponse**

```text
A container with the `unless-stopped` restart policy starts again after a daemon restart because this policy causes Docker to restart the container if it exits or if the daemon restarts, but not if you stopped the container yourself. Therefore, upon a daemon restart, the container will be restarted according to its `unless-stopped` policy unless it was manually stopped by the user.
```

**Explication du juge llama**

```text
The answer accurately describes the behavior of a container with the `unless-stopped` restart policy, including the fact that it will be restarted after a daemon restart even if the container was manually stopped. This is consistent with the information provided in the context.
```


---

## p148 | faithfulness | llama

- **id** : p148
- **split** : dev (classement exploratoire)
- **config** : E-noisy_context
- **critère** : faithfulness
- **mon verdict** : FAIL
- **llama** : note 5/5 (verdict au seuil par défaut : PASS)

**Question**

```text
Since Docker deletes anonymous volumes automatically when their container is removed, how can I keep the data they contain?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
sing your preferred tools.

## Remove volumes

A Docker data volume persists after you delete a container. There are two types
of volumes to consider:

- Named volumes have a specific name, for example, `awesome:/bar`, where `awesome` is the name.
- Anonymous volumes have no specific name. Therefore, when the container is deleted, you can instruct the Docker Engine daemon to remove them.

### Remove anonymous volumes

To automatically remove anonymous volumes, use the `--rm` option. For example,
this command creates an anonymous `/foo` volume. When you remove the container,
the Docker Engine removes the `/foo` volume but not the `awesome` volume.

The `--rm` option works with both foreground and detached (`-d`) containers.
The anonymous volumes are cleaned up when the container exits.

```console
$ docker run --rm -v /foo -v awesome:/bar busybox top
```

---

-existing files into an empty
volume, use the `volume-nocopy` option, see [Options for --mount](#options-for---mount).

## Named and anonymous volumes

A volume may be named or anonymous. Anonymous volumes are given a random name
that's guaranteed to be unique within a given Docker host. Just like named
volumes, anonymous volumes persist even if you remove the container that uses
them, except if you use the `--rm` flag when creating the container, in which
case the anonymous volume associated with the container is destroyed. See
[Remove anonymous volumes](volumes.md#remove-anonymous-volumes).

If you create multiple containers consecutively that each use anonymous
volumes, each container creates its own volume. Anonymous volumes aren't reused
or shared between containers automatically. To share an anonymous volume
between two or more containers, you must mount the anonymous volume using the
random volume ID.

## Syntax

---

`-d`) containers.
The anonymous volumes are cleaned up when the container exits.

```console
$ docker run --rm -v /foo -v awesome:/bar busybox top
```

> [!NOTE]
>
> If another container binds the volumes with
> `--volumes-from`, the volume definitions are _copied_ and the
> anonymous volume also stays after the first container is removed.

### Remove all volumes

To remove all unused volumes and free up space:

```console
$ docker volume prune
```

## Next steps

- Learn about [bind mounts](bind-mounts.md).
- Learn about [tmpfs mounts](tmpfs.md).
- Learn about [image mounts](image-mounts.md).
- Learn about [storage drivers](/engine/storage/drivers/).
- Learn about [third-party volume driver plugins](/engine/extend/legacy_plugins/).

---

threshold as
either `memory.available<10%` or `memory.available<1Gi` (you cannot use both).

#### Soft eviction thresholds {#soft-eviction-thresholds}

A soft eviction threshold pairs an eviction threshold with a required
administrator-specified grace period. The kubelet does not evict pods until the
grace period is exceeded. The kubelet returns an error on startup if you do
not specify a grace period.

You can specify both a soft eviction threshold grace period and a maximum
allowed pod termination grace period for kubelet to use during evictions. If you
specify a maximum allowed grace period and the soft eviction threshold is met,
the kubelet uses the lesser of the two grace periods. If you do not specify a
maximum allowed grace period, the kubelet kills evicted pods immediately without
graceful termination.

You can use the following flags to configure soft eviction thresholds:

---

one of its `resourceRules` or `nonResourceRules` (depending on whether the
incoming request is for a resource or non-resource URL) match the request.

For the `name` field in subjects, and the `verbs`, `apiGroups`, `resources`,
`namespaces`, and `nonResourceURLs` fields of resource and non-resource rules,
the wildcard `*` may be specified to match all values for the given field,
effectively removing it from consideration.

A FlowSchema's `distinguisherMethod.type` determines how requests matching that
schema will be separated into flows. It may be `ByUser`, in which one requesting
user will not be able to starve other users of capacity; `ByNamespace`, in which
requests for resources in one namespace will not be able to starve requests for
resources in other namespaces of capacity; or blank (or `distinguisherMethod` may be
omitted entirely), in which all requests matched by this FlowSchema will be
considered part of a single flow. The correct choice for a given FlowSchema
````

</details>

**Réponse**

```text
To keep the data contained within anonymous volumes after their associated container is removed, you would need to mount the anonymous volume into another container or manage it in a way that ensures its persistence. Since anonymous volumes are given random names and Docker does not automatically persist them across containers without explicit actions, you can:

1. **Manually Rename and Reuse**: After identifying the ID of an anonymous volume using `docker volume ls`, you could manually rename it to a named volume with a desired name using `docker volume rename`. Then use that named volume in subsequent container definitions.

2. **Use --volumes-from**: If another container needs access to the data, you can use the `--volumes-from` option to share volumes between containers.

3. **Scripted Management**: Write scripts or use Docker Compose to manage and reuse anonymous volumes across different container setups.

By taking these steps, you can effectively keep the data from anonymous volumes persistent as needed.
```

**Explication du juge llama**

```text
The answer is completely faithful to the given context. It accurately describes the behavior of Docker with anonymous volumes and provides practical steps to manage and persist their data. The answer does not introduce any unsupported claims or fabricate information, making it a faithful representation of the context provided.
```


---

## p186 | relevance | llama

- **id** : p186
- **split** : dev (classement exploratoire)
- **config** : E-noisy_context
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
What is the maximum amount of data a Kubernetes ConfigMap can hold?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
.



<!-- body -->

## Kubernetes default limits

The Kubernetes scheduler has default limits on the number of volumes
that can be attached to a Node:

<table>
  <tr><th>Cloud service</th><th>Maximum volumes per Node</th></tr>
  <tr><td><a href="https://aws.amazon.com/ebs/">Amazon Elastic Block Store (EBS)</a></td><td>39</td></tr>
  <tr><td><a href="https://cloud.google.com/persistent-disk/">Google Persistent Disk</a></td><td>16</td></tr>
  <tr><td><a href="https://azure.microsoft.com/en-us/services/storage/main-disks/">Microsoft Azure Disk Storage</a></td><td>16</td></tr>
</table>

## Dynamic volume limits

{{< feature-state state="stable" for_k8s_version="v1.17" >}}

Dynamic volume limits are supported for following volume types.

- Amazon EBS
- Google Persistent Disk
- Azure Disk
- CSI

For volumes managed by in-tree volume plugins, Kubernetes automatically determines the Node
type and enforces the appropriate maximum number of volumes for the node. For example:

---

Memory Limits
  ------------    ----------    ---------------    -------------
  680m (34%)      400m (20%)    920Mi (11%)        1070Mi (13%)
```

In the preceding output, you can see that if a Pod requests more than 1.120 CPUs
or more than 6.23Gi of memory, that Pod will not fit on the node.

By looking at the “Pods” section, you can see which Pods are taking up space on
the node.

The amount of resources available to Pods is less than the node capacity because
system daemons use a portion of the available resources. Within the Kubernetes API,
each Node has a `.status.allocatable` field
(see [NodeStatus](/docs/reference/kubernetes-api/cluster-resources/node-v1/#NodeStatus)
for details).

---

ee volume plugins, Kubernetes automatically determines the Node
type and enforces the appropriate maximum number of volumes for the node. For example:

* On
<a href="https://cloud.google.com/compute/">Google Compute Engine</a>,
up to 127 volumes can be attached to a node, [depending on the node
type](https://cloud.google.com/compute/docs/disks/#pdnumberlimits).

* For Amazon EBS disks on M5,C5,R5,T3 and Z1D instance types, Kubernetes allows only 25
volumes to be attached to a Node. For other instance types on
<a href="https://aws.amazon.com/ec2/">Amazon Elastic Compute Cloud (EC2)</a>,
Kubernetes allows 39 volumes to be attached to a Node.

* On Azure, up to 64 disks can be attached to a node, depending on the node type. For more details, refer to [Sizes for virtual machines in Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes).

---

t meet the Kubernetes [label key format](/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set).

### Status for Pod readiness

To set these `status.conditions` for the Pod, applications and
{{< glossary_tooltip term_id="operator-pattern" text="operators">}} should use
the `PATCH` action on the Pod's status subresource. You can use `kubectl patch`
with `--subresource=status`, or a [Kubernetes client library](/docs/reference/using-api/client-libraries/) to write
code that sets custom Pod conditions for Pod readiness.

For a Pod that uses custom conditions, that Pod is evaluated to be ready **only** when both the following statements apply:

- All containers in the Pod are ready.
- All conditions specified in `readinessGates` are `True`.

When a Pod's containers are Ready but at least one custom condition is missing or `False`,
the kubelet sets the Pod's `Ready` condition to `status: "False"` with `reason: ReadinessGatesNotReady`.

## {{% heading "whatsnext" %}}

---

lContainersOnContainerExits
```

A Pod requires this feature if one of its containers specifies a restart rule
with the `RestartAllContainers` action:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: example-pod
spec:
  containers:
  - name: main
    image: registry.k8s.io/busybox:1.27.2
    restartPolicy: Never     # Container restart policy must be specified if rules are specified
    restartPolicyRules:      # Restart the whole Pod in place on exit code 42
    - action: RestartAllContainers
      exitCodes:
        operator: In
        values: [42]
```

When scheduling this Pod, the kube-scheduler only considers nodes that list
`RestartAllContainersOnContainerExits` in their `.status.declaredFeatures`.
In a cluster where only some nodes have that feature gate enabled (for
example, part way through a cluster upgrade), this prevents the Pod from being
assigned to a node whose kubelet would ignore the restart rule.
````

</details>

**Réponse**

```text
The information provided in the context does not contain details about the maximum amount of data a Kubernetes ConfigMap can hold.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it does not provide any information about the maximum amount of data a Kubernetes ConfigMap can hold. The answer simply states that the context does not contain details about this topic, which is not relevant to the question asked.
```

