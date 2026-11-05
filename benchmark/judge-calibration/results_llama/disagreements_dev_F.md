# Désaccords, configuration F (broken_retrieval), split dev (classement exploratoire)

Source : `disagreements_dev_hors_seconde_passe.jsonl` (split dev (classement exploratoire), seuil par défaut 0.7, items de la seconde passe exclus). 9 entrées (item × critère), 9 désaccords juge × item × critère, dont 0 entrées où plusieurs juges sont en désaccord.

## p002 | relevance | llama

- **id** : p002
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
In FastAPI, what makes a query parameter required rather than optional?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
a"
$ contents="{...}"
$ echo $contents > contents.json
```

Make sure that the nodes to which you are deploying are correctly configured for the gMSA.

 To use a config as a credential spec, create a Docker config in a credential spec file named `credpspec.json`. 
 You can specify any name for the name of the `config`. 

```console
$ docker config create --label com.docker.gmsa.name=mygmsa credspec credspec.json
```

Now you can create a service using this credential spec. Specify the `--credential-spec` flag with the config name:

```console
$ docker service create --credential-spec="config://credspec" <your image>
```

 Your service uses the gMSA credential spec when it starts, but unlike a typical Docker config (used by passing the --config flag), the credential spec is not mounted into the container.

## Update a service

---

need to run cleanup code when a :exc:`GeneratorExit` occurs, I suggest
  using a ``try: ... finally:`` suite instead of catching :exc:`GeneratorExit`.

The cumulative effect of these changes is to turn generators from one-way
producers of information into both producers and consumers.

Generators also become **coroutines**, a more generalized form of subroutines.
Subroutines are entered at one point and exited at another point (the top of the
function, and a ``return`` statement), but coroutines can be entered, exited,
and resumed at many different points (the ``yield`` statements).


Built-in functions
==================

Let's look in more detail at built-in functions often used with iterators.

Two of Python's built-in functions, :func:`map` and :func:`filter` duplicate the
features of generator expressions:

:func:`map(f, iterA, iterB, ...) <map>` returns an iterator over the sequence
 ``f(iterA[0], iterB[0]), f(iterA[1], iterB[1]), f(iterA[2], iterB[2]), ...``.

---

er.service.d
   ```

2. Create a file named `/etc/systemd/system/docker.service.d/http-proxy.conf`
   that adds the `HTTP_PROXY` environment variable:

   ```systemd
   [Service]
   Environment="HTTP_PROXY=http://proxy.example.com:3128"
   ```

   To proxy HTTPS requests, set the `HTTPS_PROXY` environment
   variable:

   ```systemd
   [Service]
   Environment="HTTPS_PROXY=http://proxy.example.com:3128"
   ```

   Multiple environment variables can be set; to set both an HTTP and an
   HTTPS proxy;

   ```systemd
   [Service]
   Environment="HTTP_PROXY=http://proxy.example.com:3128"
   Environment="HTTPS_PROXY=http://proxy.example.com:3128"
   ```

   > [!NOTE]
   >
   > Special characters in the proxy value, such as `#?!()[]{}`, must be double
   > escaped using `%%`. For example:
   >
   > ```systemd
   > [Service]
   > Environment="HTTP_PROXY=http://domain%%5Cuser:complex%%23pass@proxy.example.com:3128/"
   > ```

---

otentially crashing the API server, but these flags are not enough to ensure
that the most important requests get through in a period of high traffic.

The API Priority and Fairness feature (APF) is an alternative that improves upon
aforementioned max-inflight limitations. APF classifies
and isolates requests in a more fine-grained way. It also introduces
a limited amount of queuing, so that no requests are rejected in cases
of very brief bursts. Requests are dispatched from queues using a
fair queuing technique so that, for example, a poorly-behaved
{{< glossary_tooltip text="controller" term_id="controller" >}} need not
starve others (even at the same priority level).

This feature is designed to work well with standard controllers, which
use informers and react to failures of API requests with exponential
back-off, and other clients that also work this way.

---

nd-line-tools-reference/feature-gates/#DRAPartitionableDevices)
feature gates to be enabled.

### Shareable summary {#resource-pool-shareable-summary}

For pools that contain [shareable devices](/docs/concepts/resource-management/dynamic-resource-allocation/dra-features/#consumable-capacity)
(devices that set `allowMultipleAllocations` and can be consumed by multiple
claims), `shareableSummary` reports aggregate capacity usage across the pool:

- `fullyAvailableDevices`: shareable devices with no capacity consumed.
- `partiallyAvailableDevices`: shareable devices with some, but not all, capacity
  consumed.
- `capacity`: per capacity name, the aggregate `total`, `consumed`, and
  `available` (`total` minus `consumed`, never negative) amounts across the pool.
````

</details>

**Réponse**

```text
The information provided in the given context does not contain any details about FastAPI or making query parameters required. Therefore, I cannot answer the question using only the provided context.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it does not contain any information about FastAPI or query parameters. It simply states that the context does not provide any relevant information, which is true but does not help answer the question.
```


---

## p025 | relevance | llama

- **id** : p025
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
In Docker Compose, does the `command` field support shell features like environment variable expansion, the way CMD does in a Dockerfile shell form?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
s command builds a Docker image named `spring-helloworld-builder` using the final stage from your `Dockerfile` file located in the current directory.


     > [!NOTE]
     >
     > In your multi-stage Dockerfile, the final stage (final) is the default target for building. This means that if you don't explicitly specify a target stage using the `--target` flag in the `docker build` command, Docker will automatically build the last stage by default. You could use `docker build -t spring-helloworld-builder --target builder .` to build only the builder stage with the JDK environment.


3. Look at the image size difference by using the `docker images` command:

    ```console
    $ docker images
    ```

    You'll get output similar to the following:

    ```console
    spring-helloworld-builder latest    c5c76cb815c0   24 minutes ago      428MB
    spring-helloworld         latest    ff708d5ee194   About an hour ago   880MB
    ```

---

e, extensible DNS server which can
  be [installed](https://github.com/coredns/helm)
  as the in-cluster DNS for pods.

## Visualization &amp; Control

* [Dashboard](https://github.com/kubernetes/dashboard#kubernetes-dashboard)
  is a dashboard web interface for Kubernetes.
* [Headlamp](https://headlamp.dev/) is an extensible Kubernetes UI that can be
  deployed in-cluster or used as a desktop application.

## Infrastructure

* [KubeVirt](https://kubevirt.io/user-guide/#/installation/installation) is an add-on
  to run virtual machines on Kubernetes. Usually run on bare-metal clusters.
* The
  [node problem detector](https://github.com/kubernetes/node-problem-detector)
  runs on Linux nodes and reports system issues as either
  [Events](/docs/reference/kubernetes-api/cluster-resources/event-v1/) or
  [Node conditions](/docs/concepts/architecture/nodes/#condition).

## Instrumentation

* [kube-state-metrics](/docs/concepts/cluster-administration/kube-state-metrics)

## Legacy Add-ons

---

Content Trust use cases, refer to the [Notary CLI documentation](https://github.com/theupdateframework/notary/blob/master/docs/command_reference.md) 

## Creating delegation keys

A prerequisite to adding your first contributor is a pair of delegation keys. 
These keys can either be generated locally using `$ docker trust`, generated by 
a certificate authority.

### Using Docker Trust to generate keys

Docker trust has a built-in generator for a delegation key pair, 
`$ docker trust generate <name>`. Running this command will automatically load 
the delegation private key in to the local Docker trust store. 

```console
$ docker trust key generate jeff

Generating key for jeff...
Enter passphrase for new jeff key with ID 9deed25: 
Repeat passphrase for new jeff key with ID 9deed25: 
Successfully generated and loaded private key. Corresponding public key available: /home/ubuntu/Documents/mytrustdir/jeff.pub
```

### Manually generating keys

---

---
reviewers:
- saad-ali
- thockin
- msau42
- jingxu97
- xing-yang
- yuxiangqian
title: Volume Snapshot Classes
content_type: concept
weight: 61 # just after volume snapshots
---

<!-- overview -->

This document describes the concept of VolumeSnapshotClass in Kubernetes. Familiarity
with [volume snapshots](/docs/concepts/storage/volume-snapshots/) and
[storage classes](/docs/concepts/storage/storage-classes) is suggested.

<!-- body -->

## Introduction

Just like StorageClass provides a way for administrators to describe the "classes"
of storage they offer when provisioning a volume, VolumeSnapshotClass provides a
way to describe the "classes" of storage when provisioning a volume snapshot.

## The VolumeSnapshotClass Resource

Each VolumeSnapshotClass contains the fields `driver`, `deletionPolicy`, and `parameters`,
which are used when a VolumeSnapshot belonging to the class needs to be
dynamically provisioned.

---

pattern described in
[managing resources](/docs/concepts/cluster-administration/manage-deployment/#canary-deployments).

## Writing a Deployment Spec

As with all other Kubernetes configs, a Deployment needs `.apiVersion`, `.kind`, and `.metadata` fields.
For general information about working with config files, see
[deploying applications](/docs/tasks/run-application/run-stateless-application-deployment/),
configuring containers, and [using kubectl to manage resources](/docs/concepts/overview/working-with-objects/object-management/) documents.
````

</details>

**Réponse**

```text
The information provided in the given context does not contain details about Docker Compose or how the `command` field supports shell features such as environment variable expansion. Therefore, I cannot answer the question based on the provided documentation.
```

**Explication du juge llama**

```text
The answer does not address the question at all. It simply states that the provided context does not contain relevant information, without attempting to answer the question. This makes the answer completely irrelevant to the question asked.
```


---

## p071 | relevance | llama

- **id** : p071
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
In FastAPI, how do path parameters and query parameters differ in how they are declared and whether they are required?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
oneB"
        p3(Pod) --> n3(Node3)
        n4(Node4)
    end
    subgraph "zoneA"
        p1(Pod) --> n1(Node1)
        p2(Pod) --> n2(Node2)
    end

    classDef plain fill:#ddd,stroke:#fff,stroke-width:4px,color:#000;
    classDef k8s fill:#326ce5,stroke:#fff,stroke-width:4px,color:#fff;
    classDef cluster fill:#fff,stroke:#bbb,stroke-width:2px,color:#326ce5;
    class n1,n2,n3,n4,p1,p2,p3 k8s;
    class zoneA,zoneB cluster;
{{< /mermaid >}}

If you want an incoming Pod to be evenly spread with existing Pods across zones, you
can use a manifest similar to:

{{% code_sample file="pods/topology-spread-constraints/one-constraint.yaml" %}}

---

g.DEBUG)
        h = BufferingSMTPHandler(options.host, options.port, options.user,
                                 options.password, options.sender,
                                 options.to, options.subject, 10)
        logger.addHandler(h)
        for i in range(102):
            logger.info("Info index = %d", i)
        h.flush()
        h.close()

If you run this script and your SMTP server is correctly set up, you should find that
it sends eleven emails to the addressee you specify. The first ten emails will each
have ten log messages, and the eleventh will have two messages. That makes up 102
messages as specified in the script.

.. _utc-formatting:

Formatting times using UTC (GMT) via configuration
--------------------------------------------------

Sometimes you want to format times using UTC, which can be done using a class
such as ``UTCFormatter``, shown below::

    import logging
    import time

---

rations can have arbitrary values.

Enumerations are Python classes, and can have methods and special methods as
usual.  If we have this enumeration::

    >>> class Mood(Enum):
    ...     FUNKY = 1
    ...     HAPPY = 3
    ...
    ...     def describe(self):
    ...         # self is the member here
    ...         return self.name, self.value
    ...
    ...     def __str__(self):
    ...         return 'my custom str! {0}'.format(self.value)
    ...
    ...     @classmethod
    ...     def favorite_mood(cls):
    ...         # cls here is the enumeration
    ...         return cls.HAPPY
    ...

Then::

    >>> Mood.favorite_mood()
    <Mood.HAPPY: 3>
    >>> Mood.HAPPY.describe()
    ('HAPPY', 3)
    >>> str(Mood.FUNKY)
    'my custom str! 1'

---

er, this does *not*
automatically give each namespace the ability to consume more resources.

Sometimes more complex policies may be desired, such as:

- Proportionally divide total cluster resources among several teams.
- Allow each tenant to grow resource usage as needed, but have a generous
  limit to prevent accidental resource exhaustion.
- Detect demand from one namespace, add nodes, and increase quota.

Such policies could be implemented using `ResourceQuotas` as building blocks, by
writing a "controller" that watches the quota usage and adjusts the quota
hard limits of each namespace according to other signals.

Note that resource quota divides up aggregate cluster resources, but it creates no
restrictions around nodes: pods from several namespaces may run on the same node.

## Quota scopes

Each quota can have an associated set of `scopes`. A quota will only measure usage for a resource if it matches
the intersection of enumerated scopes.

---

o start a container using a bind mount and map it to the container file location.

```console
$ docker run -v /HOST/PATH:/CONTAINER/PATH -it nginx
```

The `--mount` flag offers more advanced features and granular control, making it suitable for complex mount scenarios or production deployments. By default, if you use `--mount` to bind-mount a file or directory that doesn't yet exist on the Docker host, the `docker run` command doesn't automatically create it for you but generates an error.

```console
$ docker run --mount type=bind,source=/HOST/PATH,target=/CONTAINER/PATH,readonly nginx
```

> [!NOTE]
>
> Docker recommends using the `--mount` syntax instead of `-v`. It provides better control over the mounting process and avoids potential issues with missing directories.

### File permissions for Docker access to host files
````

</details>

**Réponse**

```text
The provided context does not contain any information about FastAPI or the differences between path parameters and query parameters. Therefore, the necessary details to answer the question cannot be derived from the given documentation.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it states that the necessary details to answer the question cannot be derived from the given documentation. This implies that the answer is completely irrelevant to the question asked.
```


---

## p074 | relevance | llama

- **id** : p074
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
What is the difference between Docker volumes and bind mounts, and why are volumes the recommended option for persisting container data?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
name: job-1
spec:
  priorityClassName: high-priority
```

### CompositePodGroup priority

{{< feature-state feature_gate_name="CompositePodGroup" >}}

`CompositePodGroup` API has `priorityClassName` and `priority` fields as well and their resolution
is performed in the same way as for the `PodGroups`, through the priority admission controller.

The priority of a root `CompositePodGroup` acts as the authoritative priority for all child groups
and Pods within its hierarchy during
[workload-aware preemption](/docs/concepts/scheduling-eviction/workload-aware-preemption/) events.
All Pods within a single group hierarchy must share the exact same priority and must be equal to the
priority of the root `CompositePodGroup`.

The value of priority is also used for the ordering of root `CompositePodGroups` in the scheduling
active queue.

---

argument (``voltage``) and three optional arguments
(``state``, ``action``, and ``type``).  This function can be called in any
of the following ways::

   parrot(1000)                                          # 1 positional argument
   parrot(voltage=1000)                                  # 1 keyword argument
   parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
   parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
   parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
   parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword

but all the following calls would be invalid::

   parrot()                     # required argument missing
   parrot(voltage=5.0, 'dead')  # non-keyword argument after a keyword argument
   parrot(110, voltage=220)     # duplicate value for the same argument
   parrot(actor='John Cleese')  # unknown keyword argument

---

es.
You can find more information about `ExternalName` resolution in
[DNS for Services and Pods](/docs/concepts/services-networking/dns-pod-service/).

<!-- preserve existing hyperlinks -->
<a id="shortcomings" />
<a id="the-gory-details-of-virtual-ips" />
<a id="proxy-modes" />
<a id="proxy-mode-userspace" />
<a id="proxy-mode-iptables" />
<a id="proxy-mode-ipvs" />
<a id="ips-and-vips" />

## Virtual IP addressing mechanism

Read [Virtual IPs and Service Proxies](/docs/reference/networking/virtual-ips/) explains the
mechanism Kubernetes provides to expose a Service with a virtual IP address.

### Traffic policies

You can set the `.spec.internalTrafficPolicy` and `.spec.externalTrafficPolicy` fields
to control how Kubernetes routes traffic to healthy (“ready”) backends.

See [Traffic Policies](/docs/reference/networking/virtual-ips/#traffic-policies) for more details.

### Traffic distribution control {#traffic-distribution}

---

It is available through the `hostname` command or the
[`gethostname`](https://man7.org/linux/man-pages/man2/gethostname.2.html)
function call in libc.

The Pod name and namespace are available as environment variables through the
[downward API](/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/).

User-defined environment variables from the Pod definition are also available to the Container,
as are any environment variables specified statically in the container image.

### Cluster information

A list of all services that were running when a Container was created is available to that Container as environment variables.
This list is limited to services within the same namespace as the new Container's Pod and Kubernetes control plane services.

For a service named *foo* that exposes a set of Pods, each running a container named *bar*,
the following variables are defined:

---

e Pod or container can also apply to pages in memory backed
  volumes, such as an `emptyDir`. The kubelet tracks `tmpfs` emptyDir volumes as container
  memory use, rather than as local [ephemeral storage](/docs/concepts/storage/ephemeral-storage/).
  When using memory backed `emptyDir`,
  be sure to check the notes [below](#memory-backed-emptydir).

If a container exceeds its memory request and the node that it runs on becomes short of
memory overall, it is likely that the Pod the container belongs to will be
{{< glossary_tooltip text="evicted" term_id="eviction" >}}.

A container might or might not be allowed to exceed its CPU limit for extended periods of time.
However, container runtimes don't terminate Pods or containers for excessive CPU usage.

To determine whether a container cannot be scheduled or is being killed due to resource limits,
see the [Troubleshooting](#troubleshooting) section.

### Resizing container resources
````

</details>

**Réponse**

```text
The provided context does not contain information about Docker volumes, bind mounts, or their differences. Therefore, the information needed to answer the question is not present in the given documentation.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it states that the context does not contain information about Docker volumes and bind mounts, which is the main topic of the question.
```


---

## p108 | relevance | llama

- **id** : p108
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
Besides Swagger UI, FastAPI serves an alternative documentation UI. At which URL is it available when running locally?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
nserts a ``\uNNNN`` escape sequence) and
``namereplace`` (inserts a ``\N{...}`` escape sequence).

The following example shows the different results::

    >>> u = chr(40960) + 'abcd' + chr(1972)
    >>> u.encode('utf-8')
    b'\xea\x80\x80abcd\xde\xb4'
    >>> u.encode('ascii')  #doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    UnicodeEncodeError: 'ascii' codec can't encode character '\ua000' in
      position 0: ordinal not in range(128)
    >>> u.encode('ascii', 'ignore')
    b'abcd'
    >>> u.encode('ascii', 'replace')
    b'?abcd?'
    >>> u.encode('ascii', 'xmlcharrefreplace')
    b'&#40960;abcd&#1972;'
    >>> u.encode('ascii', 'backslashreplace')
    b'\\ua000abcd\\u07b4'
    >>> u.encode('ascii', 'namereplace')
    b'\\N{YI SYLLABLE IT}abcd\\u07b4'

---

# Body - Fields { #body-fields }

The same way you can declare additional validation and metadata in *path operation function* parameters with `Query`, `Path` and `Body`, you can declare validation and metadata inside of Pydantic models using Pydantic's `Field`.

## Import `Field` { #import-field }

First, you have to import it:

{* ../../docs_src/body_fields/tutorial001_an_py310.py hl[4] *}


/// warning

Notice that `Field` is imported directly from `pydantic`, not from `fastapi` as are all the rest (`Query`, `Path`, `Body`, etc).

///

## Declare model attributes { #declare-model-attributes }

You can then use `Field` with model attributes:

{* ../../docs_src/body_fields/tutorial001_an_py310.py hl[11:14] *}

`Field` works the same way as `Query`, `Path` and `Body`, it has all the same parameters, etc.

/// note | Technical Details

---

orms below are equivalent:

```yml
test: ["CMD-SHELL", "curl -f http://localhost || exit 1"]
```

```yml
test: curl -f https://localhost || exit 1
```

`NONE` disables the healthcheck, and is mostly useful to disable the Healthcheck Dockerfile instruction set by the service's Docker image. Alternatively,
the healthcheck set by the image can be disabled by setting `disable: true`:

```yml
healthcheck:
  disable: true
```

### hostname

`hostname` declares a custom host name to use for the service container. It must be a valid RFC 1123 hostname.

### image

`image` specifies the image to start the container from. `image` must follow the Open Container Specification
[addressable image format](https://github.com/opencontainers/org/blob/master/docs/docs/introduction/digests.md),
as `[<registry>/][<project>/]<image>[:<tag>|@<digest>]`.

---

.2 behaviour, ``logging.lastResort`` can be set to ``None``.

.. _library-config:

Configuring Logging for a Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When developing a library which uses logging, you should take care to
document how the library uses logging - for example, the names of loggers
used. Some consideration also needs to be given to its logging configuration.
If the using application does not use logging, and library code makes logging
calls, then (as described in the previous section) events of severity
``WARNING`` and greater will be printed to ``sys.stderr``. This is regarded as
the best default behaviour.

---

* Dispatch to multiple subcommands in separate files, all logging at the same
  level in a consistent way
* Make use of simple, minimal configuration

Suppose we have a command-line application whose job is to stop, start or
restart some services. This could be organised for the purposes of illustration
as a file ``app.py`` that is the main script for the application, with individual
commands implemented in ``start.py``, ``stop.py`` and ``restart.py``. Suppose
further that we want to control the verbosity of the application via a
command-line argument, defaulting to ``logging.INFO``. Here's one way that
``app.py`` could be written::

    import argparse
    import importlib
    import logging
    import os
    import sys
````

</details>

**Réponse**

```text
The information provided in the context does not contain an answer to the question about the alternative documentation UI for FastAPI when running locally. Therefore, I cannot provide an answer based on the given context alone.
```

**Explication du juge llama**

```text
The answer provided does not address the question about the alternative documentation UI for FastAPI when running locally, making it completely irrelevant to the question asked.
```


---

## p110 | relevance | llama

- **id** : p110
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
How many Uvicorn workers does the FastAPI documentation recommend running in production?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
(Node3)
        p5(Pod) --> n3
    end
    subgraph "zoneA"
        p1(Pod) --> n1(Node1)
        p2(Pod) --> n1
        p3(Pod) --> n2(Node2)
    end

    classDef plain fill:#ddd,stroke:#fff,stroke-width:4px,color:#000;
    classDef k8s fill:#326ce5,stroke:#fff,stroke-width:4px,color:#fff;
    classDef cluster fill:#fff,stroke:#bbb,stroke-width:2px,color:#326ce5;
    class n1,n2,n3,n4,p1,p2,p3,p4,p5 k8s;
    class zoneA,zoneB cluster;
{{< /mermaid >}}

If you were to apply
[`two-constraints.yaml`](https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/pods/topology-spread-constraints/two-constraints.yaml)
(the manifest from the previous example)
to **this** cluster, you would see that the Pod `mypod` stays in the `Pending` state.
This happens because: to satisfy the first constraint, the Pod `mypod` can only
be placed into zone `B`; while in terms of the second constraint, the Pod `mypod`

---

reuse [Kubernetes API support features](#common-features).  | You don't need those features. |

### Declarative APIs

In a Declarative API, typically:

- Your API consists of a relatively small number of relatively small objects (resources).
- The objects define configuration of applications or infrastructure.
- The objects are updated relatively infrequently.
- Humans often need to read and write the objects.
- The main operations on the objects are CRUD-y (creating, reading, updating and deleting).
- Transactions across objects are not required: the API represents a desired state, not an exact state.

Imperative APIs are not declarative.
Signs that your API might not be declarative include:

---

cycle succeeds, the condition is set to `True` with reason
`Scheduled`. For `gang` policy PodGroups, this means at least `minCount` Pods were
placed.

When scheduling fails, the condition is set to `False` with one of the following
reasons:

* `Unschedulable` — the group could not be placed due to resource constraints,
  affinity or anti-affinity rules, or insufficient capacity for the gang.
* `SchedulerError` — scheduling failed because of an internal scheduler error
  (for example, while parsing scheduling constraints such as `nodeAffinity`).

Once this condition is set to `True`, it never changes.

You can check conditions with:

```shell
kubectl get podgroup <name> -o jsonpath='{.status.conditions}'
```

## CompositePodGroup conditions

{{< feature-state feature_gate_name="CompositePodGroup" >}}

CompositePodGroup API exposes the `status.conditions` field as well.

In v1.37, the scheduler does not populate this field, however.

## {{% heading "whatsnext" %}}

---

u've decided to undo the current rollout and rollback to the previous revision:
   ```shell
   kubectl rollout undo deployment/nginx-deployment
   ```

   The output is similar to this:
   ```
   deployment.apps/nginx-deployment rolled back
   ```
   Alternatively, you can rollback to a specific revision by specifying it with `--to-revision`:

   ```shell
   kubectl rollout undo deployment/nginx-deployment --to-revision=2
   ```

   The output is similar to this:
   ```
   deployment.apps/nginx-deployment rolled back
   ```

   For more details about rollout related commands, read [`kubectl rollout`](/docs/reference/generated/kubectl/kubectl-commands#rollout).

   The Deployment is now rolled back to a previous stable revision. As you can see, a `DeploymentRollback` event
   for rolling back to revision 2 is generated from Deployment controller.

---

le resources and it does not
  detail specific schema for the resources. For reference about resource schemas,
  please refer to the OpenAPI document.

- The [Kubernetes OpenAPI Document](#openapi-interface-definition) provides (full)
  [OpenAPI v2.0 and 3.0 schemas](https://www.openapis.org/) for all Kubernetes API
endpoints.
  The OpenAPI v3 is the preferred method for accessing OpenAPI as it
provides
  a more comprehensive and accurate view of the API. It includes all the available
  API paths, as well as all resources consumed and produced for every operations
  on every endpoints. It also includes any extensibility components that a cluster supports.
  The data is a complete specification and is significantly larger than that from the
  Discovery API.

## Discovery API

Kubernetes publishes a list of all group versions and resources supported via
the Discovery API. This includes the following for each resource:
````

</details>

**Réponse**

```text
The information provided in the given context does not contain any details about Uvicorn workers or FastAPI documentation recommendations for production environments. Therefore, I cannot provide an answer to the question based on the given context.
```

**Explication du juge llama**

```text
The answer explicitly states that it cannot provide an answer to the question because the given context does not contain any relevant information. This indicates that the answer is completely irrelevant to the question asked.
```


---

## p122 | relevance | llama

- **id** : p122
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
Which Python version does venv install into a new virtual environment?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
/magicstorage
parameters:
  resturl: "http://192.168.10.100:8080"
  restuser: ""
  secretNamespace: ""
  secretName: ""
allowVolumeExpansion: true
```

To request a larger volume for a PVC, edit the PVC object and specify a larger
size. This triggers expansion of the volume that backs the underlying PersistentVolume. A
new PersistentVolume is never created to satisfy the claim. Instead, an existing volume is resized.

{{< warning >}}
Directly editing the size of a PersistentVolume can prevent an automatic resize of that volume.
If you edit the capacity of a PersistentVolume, and then edit the `.spec` of a matching
PersistentVolumeClaim to make the size of the PersistentVolumeClaim match the PersistentVolume,
then no storage resize happens.
The Kubernetes control plane will see that the desired state of both resources matches,
conclude that the backing volume size has been manually
increased and that no resize is necessary.
{{< /warning >}}

#### CSI Volume expansion

---

configured to be expandable. This allows you to resize the
volume by editing the corresponding PVC object, requesting a new larger amount of
storage.

The following types of volumes support volume expansion, when the underlying
StorageClass has the field `allowVolumeExpansion` set to true.

{{< table caption = "Table of Volume types and the version of Kubernetes they require"  >}}

| Volume type          | Required Kubernetes version for volume expansion |
| :------------------- | :----------------------------------------------- |
| Azure File           | 1.11                                             |
| CSI                  | 1.24                                             |
| FlexVolume           | 1.13                                             |
| Portworx             | 1.11                                             |
| rbd                  | 1.11                                             |

{{< /table >}}

---

| `--log-opt compress=true`                          |

### Examples

This example starts an `alpine` container which can have a maximum of 3 log
files no larger than 10 megabytes each.

```console
$ docker run -it --log-opt max-size=10m --log-opt max-file=3 alpine ash
```

---

er_name`, `status`, and `reason`, with a value of `1` for each storage-backend
condition currently reported for a driver on that node.

## Limitations

- Kubernetes only surfaces these health reports; it does not act on them.
  Nothing in Kubernetes reschedules Pods, fails over volumes, or otherwise reacts
  to a reported condition on its own. Building a remediation controller on top of
  these status fields is left to cluster operators and vendors.
- An older, alpha implementation of the same `CSIVolumeHealth` feature gate (available
  starting in Kubernetes v1.21) reported abnormal volume conditions using Kubernetes
  Events and a `kubelet_volume_stats_health_status_abnormal` metric. That mechanism
  has been replaced by the status fields and RPCs described on this page, and no
  longer exists.

## {{% heading "whatsnext" %}}

---

to expose a single Service
(see [alternatives](#alternatives)). You can also do this with an Ingress by specifying a
*default backend* with no rules.

{{% code_sample file="service/networking/test-ingress.yaml" %}}

If you create it using `kubectl apply -f` you should be able to view the state
of the Ingress you added:

```bash
kubectl get ingress test-ingress
```

```
NAME           CLASS         HOSTS   ADDRESS         PORTS   AGE
test-ingress   external-lb   *       203.0.113.123   80      59s
```

Where `203.0.113.123` is the IP allocated by the Ingress controller to satisfy
this Ingress.

{{< note >}}
Ingress controllers and load balancers may take a minute or two to allocate an IP address.
Until that time, you often see the address listed as `<pending>`.
{{< /note >}}

### Simple fanout
````

</details>

**Réponse**

```text
The information needed to answer the question about which Python version `venv` installs into a new virtual environment is not present in the provided documentation.
```

**Explication du juge llama**

```text
The answer states that the information needed to answer the question is not present in the provided documentation, which means it does not provide any relevant information about which Python version `venv` installs into a new virtual environment. This makes the answer completely irrelevant to the question asked.
```


---

## p132 | faithfulness | llama

- **id** : p132
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : faithfulness
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
I manually stopped a container that has the restart policy `always`. What happens to it when the Docker daemon restarts?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
-affinity)
- [nodeName](#nodename) field
- [Pod topology spread constraints](#pod-topology-spread-constraints)

## Node labels {#built-in-node-labels}

Like many other Kubernetes objects, nodes have
[labels](/docs/concepts/overview/working-with-objects/labels/). You can
[attach labels manually](/docs/tasks/configure-pod-container/assign-pods-nodes/#add-a-label-to-a-node).
Kubernetes also populates a [standard set of labels](/docs/reference/node/node-labels/)
on all nodes in a cluster.

{{<note>}}
The value of these labels is cloud provider specific and is not guaranteed to be reliable.
For example, the value of `kubernetes.io/hostname` may be the same as the node name in some environments
and a different value in other environments.
{{</note>}}

### Node isolation/restriction

---

e following command:

```shell
kubectl apply -f https://k8s.io/examples/debug/counter-pod.yaml
```

The output is:

```console
pod/counter created
```

To fetch the logs, use the `kubectl logs` command, as follows:

```shell
kubectl logs counter
```

The output is similar to:

```console
0: Fri Apr  1 11:42:23 UTC 2022
1: Fri Apr  1 11:42:24 UTC 2022
2: Fri Apr  1 11:42:25 UTC 2022
```

You can use `kubectl logs --previous` to retrieve logs from a previous instantiation of a container.
If your pod has multiple containers, specify which container's logs you want to access by
appending a container name to the command, with a `-c` flag, like so:

```shell
kubectl logs counter -c count
```


### Container log streams

{{< feature-state feature_gate_name="PodLogsQuerySplitStreams" >}}

---

e-scheduler
should use its compiled-in default.
If you set `percentageOfNodesToScore` above 100, kube-scheduler acts as if you
had set a value of 100.

To change the value, edit the
[kube-scheduler configuration file](/docs/reference/config-api/kube-scheduler-config.v1/)
and then restart the scheduler.
In many cases, the configuration file can be found at `/etc/kubernetes/config/kube-scheduler.yaml`.

After you have made this change, you can run

```bash
kubectl get pods -n kube-system | grep kube-scheduler
```

to verify that the kube-scheduler component is healthy.

## Node scoring threshold {#percentage-of-nodes-to-score}

To improve scheduling performance, the kube-scheduler can stop looking for
feasible nodes once it has found enough of them. In large clusters, this saves
time compared to a naive approach that would consider every node.

---

lude <abstractions/base>

  network inet tcp,
  network inet udp,
  network inet icmp,

  deny network raw,

  deny network packet,

  file,
  umount,

  deny /bin/** wl,
  deny /boot/** wl,
  deny /dev/** wl,
  deny /etc/** wl,
  deny /home/** wl,
  deny /lib/** wl,
  deny /lib64/** wl,
  deny /media/** wl,
  deny /mnt/** wl,
  deny /opt/** wl,
  deny /proc/** wl,
  deny /root/** wl,
  deny /sbin/** wl,
  deny /srv/** wl,
  deny /tmp/** wl,
  deny /sys/** wl,
  deny /usr/** wl,

  audit /** w,

  /var/run/nginx.pid w,

  /usr/sbin/nginx ix,

  deny /bin/dash mrwklx,
  deny /bin/sh mrwklx,
  deny /usr/bin/top mrwklx,


  capability chown,
  capability dac_override,
  capability setuid,
  capability setgid,
  capability net_bind_service,

---

yet exist, create it. Assuming
    that the file was empty, add the following contents.

    ```json
    {
      "storage-driver": "vfs"
    }
    ```

    If you want to set a quota to control the maximum size the VFS storage
    driver can use, set the `size` option on the `storage-opts` key.

    ```json
    {
      "storage-driver": "vfs",
      "storage-opts": ["size=256M"]
    }
    ```

    Docker doesn't start if the `daemon.json` file contains invalid JSON.

3.  Start Docker.

    ```console
    $ sudo systemctl start docker
    ```

4.  Verify that the daemon is using the `vfs` storage driver.
    Use the `docker info` command and look for `Storage Driver`.

    ```console
    $ docker info

    Storage Driver: vfs
    ...
    ```

Docker is now using the `vfs` storage driver. Docker has automatically
created the `/var/lib/docker/vfs/` directory, which contains all the layers
used by running containers.

## How the `vfs` storage driver works
````

</details>

**Réponse**

```text
The provided context does not contain information about what happens to a container with a restart policy of `always` when the Docker daemon restarts after being stopped. Therefore, the information needed to answer this specific question is not present in the given documentation.
```

**Explication du juge llama**

```text
The provided context does not contain any information about what happens to a container with a restart policy of `always` when the Docker daemon restarts after being stopped. Therefore, the answer is completely unfaithful to the given context.
```


---

## p209 | relevance | llama

- **id** : p209
- **split** : dev (classement exploratoire)
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
What command do I run to create a virtual environment called tutorial-env?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
of the image is running and more difficult to
roll back properly.

Instead, specify a meaningful tag such as `v1.42.0` and/or a digest.
{{< /note >}}

To make sure the Pod always uses the same version of a container image, you can specify
the image's digest;
replace `<image-name>:<tag>` with `<image-name>@<digest>`
(for example, `image@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2`).

When using image tags, if the image registry were to change the code that the tag on that image
represents, you might end up with a mix of Pods running the old and new code. An image digest
uniquely identifies a specific version of the image, so Kubernetes runs the same code every time
it starts a container with that image name and digest specified. Specifying an image by digest
pins the code that you run so that a change at the registry cannot lead to that mix of versions.

---

m the :mod:`argparse`
module and output them into a file named ``messages.po``. This command assumes
that your Python installation is in ``/usr/lib``.

You can find out the location of the :mod:`argparse` module on your system
using this script::

   import argparse
   print(argparse.__file__)

Once the messages in the ``.po`` file are translated and the translations are
installed using :mod:`gettext`, :mod:`argparse` will be able to display the
translated messages.

To translate your own strings in the :mod:`argparse` output, use :mod:`gettext`.

Conclusion
==========

The :mod:`argparse` module offers a lot more than shown here.
Its docs are quite detailed and thorough, and full of examples.
Having gone through this tutorial, you should easily digest them
without feeling overwhelmed.

---

h this setup, the containers write their logs to separate subdirectories of
the `logs` volume. The containers can't access the other container's logs.

## Share data between machines

When building fault-tolerant applications, you may need to configure multiple
replicas of the same service to have access to the same files.

![shared storage](images/volumes-shared-storage.webp)

There are several ways to achieve this when developing your applications.
One is to add logic to your application to store files on a cloud object
storage system like Amazon S3. Another is to create volumes with a driver that
supports writing files to an external storage system like NFS or Amazon S3.

Volume drivers let you abstract the underlying storage system from the
application logic. For example, if your services use a volume with an NFS
driver, you can update the services to use a different driver. For example, to
store data in the cloud, without changing the application logic.

## Use a volume driver

---

nect to MySQL.

    First, find the ID of the `mysql` container task.

    ```console
    $ docker ps --filter name=mysql -q

    c7705cf6176f
    ```

    Substitute the ID in the command below, or use the second variant which
    uses shell expansion to do it all in a single step.

    ```console
    $ docker container exec <CONTAINER_ID> \
        bash -c 'mysqladmin --user=wordpress --password="$(< /run/secrets/old_mysql_password)" password "$(< /run/secrets/mysql_password)"'
    ```

    Or:

    ```console
    $ docker container exec $(docker ps --filter name=mysql -q) \
        bash -c 'mysqladmin --user=wordpress --password="$(< /run/secrets/old_mysql_password)" password "$(< /run/secrets/mysql_password)"'
    ```

4.  Update the `wordpress` service to use the new password, keeping the target
    path at `/run/secrets/wp_db_password`. This triggers a rolling restart of
    the WordPress service and the new secret is used.

---

etes {{< glossary_tooltip term_id="service" >}} that identifies
  a set of Pods using {{< glossary_tooltip text="label" term_id="label" >}} selectors.
  Unless mentioned otherwise, Services are assumed to have virtual IPs only routable within the cluster network.

## What is Ingress?

[Ingress](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/#ingress-v1-networking-k8s-io)
exposes HTTP and HTTPS routes from outside the cluster to
{{< link text="services" url="/docs/concepts/services-networking/service/" >}} within the cluster.
Traffic routing is controlled by rules defined on the Ingress resource.

Here is a simple example where an Ingress sends all its traffic to one Service:
````

</details>

**Réponse**

```text
The provided context does not contain information about creating a virtual environment called tutorial-env. Therefore, the answer cannot be derived from the given documentation.
```

**Explication du juge llama**

```text
The answer does not address the question at all, as it states that the context does not contain information about creating a virtual environment called tutorial-env. This is a direct contradiction to the question asked, which specifically asks for the command to create such an environment. Therefore, the answer is completely irrelevant to the question.
```

