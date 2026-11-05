# Désaccords, configuration F (broken_retrieval), split test

Source : `disagreements_hors_seconde_passe.jsonl` (split test, seuil par défaut 0.7, items de la seconde passe exclus). 4 entrées (item × critère), 4 désaccords juge × item × critère, dont 0 entrées où plusieurs juges sont en désaccord.

## p012 | relevance | llama

- **id** : p012
- **split** : test
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
In a Docker Compose file, what is the build context if I don't set one explicitly?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
scaling the number of replicas up or down, either manually or by an auto-scaling control agent, by updating the `replicas` field.

### Rolling updates

The ReplicationController is designed to facilitate rolling updates to a service by replacing pods one-by-one.

As explained in [#1353](https://issue.k8s.io/1353), the recommended approach is to create a new ReplicationController with 1 replica, scale the new (+1) and old (-1) controllers one by one, and then delete the old controller after it reaches 0 replicas. This predictably updates the set of pods regardless of unexpected failures.

Ideally, the rolling update controller would take application readiness into account, and would ensure that a sufficient number of pods were productively serving at any given time.

---

e same time.

## Enable embedded containerd

Add the `embedded-containerd` feature to the
[daemon configuration file](./_index.md#configuration-file):

```json
{
  "features": {
    "embedded-containerd": true
  }
}
```

Restart the Docker daemon:

```console
$ sudo systemctl restart docker
```

The `embedded-containerd` feature takes precedence over a containerd address
set with the `--containerd` daemon flag. This behavior means that packaged
service configurations that set a containerd address don't prevent embedded
containerd from starting.

> [!IMPORTANT]
>
> Embedded containerd keeps its state under the Docker data root, in
> `/var/lib/docker/containerd/daemon` by default. A containerd installed on the
> host keeps its own state elsewhere, such as `/var/lib/containerd`. If the
> daemon used such a containerd before, the containers and images stored there
> aren't available in embedded mode.

You can also enable the feature when starting `dockerd` manually:

---

### Docker and iptables chains

To support bridge and overlay networks, Docker creates the following custom
`iptables` chains in the `filter` table:

* `DOCKER-USER`
    * A placeholder for user-defined rules that will be processed before rules
      in the `DOCKER-FORWARD` and `DOCKER` chains.
* `DOCKER-FORWARD`
    * The first stage of processing for Docker's networks. Rules that pass packets
      that are not related to established connections to the other Docker chains,
      as well as rules to accept packets that are part of established connections.
* `DOCKER`, `DOCKER-BRIDGE`, `DOCKER-INTERNAL`
    * Rules that determine whether a packet that is not part of an established 
      connection should be accepted, based on the port forwarding configuration
      of running containers.
* `DOCKER-CT`
    * Per-bridge connection tracking rules.
* `DOCKER-INGRESS`
    * Rules related to Swarm networking.

---

ation: 0
            score: 0
          - utilization: 100
            score: 10
        type: RequestedToCapacityRatio
    name: NodeResourcesFit
```

In this example, only the extended resources `intel.com/foo` and `intel.com/bar` are listed in
`resources`. The `NodeResourcesFit` plugin therefore scores nodes based solely on the utilization
of those two resources; CPU and memory do not contribute to the score from this plugin. Because the
configured shape assigns a higher score as utilization increases (`score: 0` at `utilization: 0`
rising to `score: 10` at `utilization: 100`), the scheduler prefers nodes where more of these
extended resources are already in use, bin-packing requests for them onto as few nodes as possible.

To include CPU and memory in this scoring strategy, add them to the `resources` list. Note that all
resources in the list share the same `shape` function, so doing so will apply the same bin-packing
curve to those resources as well.

---

------------:|
| pod-a  *terminating* | pod-b *available*   | pod-c *available*  |
| pod-x  *terminating* |                     |                    |

The deployment notices that one of the pods is terminating, so it creates a replacement
called `pod-d`.  Since `node-1` is cordoned, it lands on another node.  Something has
also created `pod-y` as a replacement for `pod-x`.

(Note: for a StatefulSet, `pod-a`, which would be called something like `pod-0`, would need
to terminate completely before its replacement, which is also called `pod-0` but has a
different UID, could be created.  Otherwise, the example applies to a StatefulSet as well.)

Now the cluster is in this state:

|   node-1 *draining*  |       node-2        |       node-3       |
|:--------------------:|:-------------------:|:------------------:|
| pod-a  *terminating* | pod-b *available*   | pod-c *available*  |
| pod-x  *terminating* | pod-d *starting*    | pod-y              |
````

</details>

**Réponse**

```text
The provided context does not contain information about Docker Compose or build contexts. Therefore, the information needed to answer the question is not present in the provided documentation.
```

**Explication du juge llama**

```text
The answer does not address the question at all. It mentions that the provided context does not contain information about Docker Compose or build contexts, but this is irrelevant to the question being asked. The question specifically asks about the build context in a Docker Compose file if one is not set explicitly, but the answer does not provide any information about this.
```


---

## p033 | relevance | llama

- **id** : p033
- **split** : test
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
Is the Kubernetes Endpoints API deprecated?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
g, and to see
which logger methods to use when, see the table below. It states, for each of a
set of common tasks, the best tool to use for that task.

+-------------------------------------+--------------------------------------+
| Task you want to perform            | The best tool for the task           |
+=====================================+======================================+
| Display console output for ordinary | :func:`print`                        |
| usage of a command line script or   |                                      |
| program                             |                                      |
+-------------------------------------+--------------------------------------+
| Report events that occur during     | A logger's :meth:`~Logger.info` (or  |
| normal operation of a program (e.g. | :meth:`~Logger.debug` method for very|
| for status monitoring or fault      | detailed output for diagnostic       |

---

---
title: Ingress Controllers
description: >-
  In order for an [Ingress](/docs/concepts/services-networking/ingress/) to work in your cluster,
  there must be an _ingress controller_ running.
  You need to select at least one ingress controller and make sure it is set up in your cluster.  
  This page lists common ingress controllers that you can deploy.
content_type: concept
weight: 50
---

{{< note >}}
The Kubernetes project recommends using [Gateway](https://gateway-api.sigs.k8s.io/) instead of
[Ingress](/docs/concepts/services-networking/ingress/).
The Ingress API has been frozen.

This means that:
* The Ingress API is generally available, and is subject to the [stability guarantees](/docs/reference/using-api/deprecation-policy/#deprecating-parts-of-the-api) for generally available APIs.
  The Kubernetes project has no plans to remove Ingress from Kubernetes.
* The Ingress API is no longer being developed, and will have no further changes
  or updates made to it.
{{< /note >}}

---

nly layers.
  - If you create and then delete a file or directory within the container's
    writable layer, the blocks are reclaimed by the `zpool`.


## ZFS and Docker performance

There are several factors that influence the performance of Docker using the
`zfs` storage driver.

- **Memory**: Memory has a major impact on ZFS performance. ZFS was originally
  designed for large enterprise-grade servers with a large amount of memory.

- **ZFS Features**: ZFS includes a de-duplication feature. Using this feature
  may save disk space, but uses a large amount of memory. It is recommended that
  you disable this feature for the `zpool` you are using with Docker, unless you
  are using SAN, NAS, or other hardware RAID technologies.

---

aml" %}}

After creating the Ingress above, you can view it with the following command:

```bash
kubectl describe ingress ingress-resource-backend
```

```
Name:             ingress-resource-backend
Namespace:        default
Address:
Default backend:  APIGroup: k8s.example.com, Kind: StorageBucket, Name: static-assets
Rules:
  Host        Path  Backends
  ----        ----  --------
  *
              /icons   APIGroup: k8s.example.com, Kind: StorageBucket, Name: icon-assets
Annotations:  <none>
Events:       <none>
```

### Path types

Each path in an Ingress is required to have a corresponding path type. Paths
that do not include an explicit `pathType` will fail validation. There are three
supported path types:

* `ImplementationSpecific`: With this path type, matching is up to the
  IngressClass. Implementations can treat this as a separate `pathType` or treat
  it identically to `Prefix` or `Exact` path types.

* `Exact`: Matches the URL path exactly and with case sensitivity.

---

for the service to be
connected to the `default` network:

```yml
services:
  some-service:
    image: foo
```
This example is actually equivalent to:

```yml
services:
  some-service:
    image: foo
    networks:
      default: {}
```

If you want the service to not be connected a network, you must set [`network_mode: none`](#network_mode).

#### `aliases`

`aliases` declares alternative hostnames for the service on the network. Other containers on the same
network can use either the service name or an alias to connect to one of the service's containers.

Since `aliases` are network-scoped, the same service can have different aliases on different networks.

> [!NOTE]
> A network-wide alias can be shared by multiple containers, and even by multiple services.
> If it is, then exactly which container the name resolves to is not guaranteed.
````

</details>

**Réponse**

```text
The information provided in the given context does not mention whether the Kubernetes Endpoints API is deprecated or not. Therefore, based on the provided documentation, I cannot answer this question.
```

**Explication du juge llama**

```text
The answer explicitly states that it cannot answer the question, which means it does not address the main point of the question at all.

The answer is completely irrelevant to the question asked, as it does not provide any information about the Kubernetes Endpoints API being deprecated or not. It simply states that the provided documentation does not mention it, which is not a relevant answer to the question.
```


---

## p105 | relevance | llama

- **id** : p105
- **split** : test
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
When I run `from module import *` in Python, why do names starting with an underscore get imported along with everything else?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
ult is 30 seconds.
Once a pod has become ready, it considers any transition to
ready to be the first if it occurred within a longer, configurable time
since it started. This value is configured with the
`--horizontal-pod-autoscaler-cpu-initialization-period` command line option,
and its default is 5 minutes.

The \\( currentMetricValue \over desiredMetricValue \\) base scale ratio is then
calculated, using the remaining pods not set aside or discarded from above.

If there were any missing metrics, the control plane recomputes the average more
conservatively, assuming those pods were consuming 100% of the desired
value in case of a scale down, and 0% in case of a scale up. This dampens
the magnitude of any potential scale.

---

e details.


Using file rotation
-------------------

.. sectionauthor:: Doug Hellmann, Vinay Sajip (changes)
.. (see <https://pymotw.com/3/logging/>)

Sometimes you want to let a log file grow to a certain size, then open a new
file and log to that. You may want to keep a certain number of these files, and
when that many files have been created, rotate the files so that the number of
files and the size of the files both remain bounded. For this usage pattern, the
logging package provides a :class:`RotatingFileHandler`::

   import glob
   import logging
   import logging.handlers

   LOG_FILENAME = 'logging_rotatingfile_example.out'

   # Set up a specific logger with our desired output level
   my_logger = logging.getLogger('MyLogger')
   my_logger.setLevel(logging.DEBUG)

   # Add the log message handler to the logger
   handler = logging.handlers.RotatingFileHandler(
                 LOG_FILENAME, maxBytes=20, backupCount=5)

   my_logger.addHandler(handler)

---

``shell
kubectl get SampleDB                   # find configured databases

kubectl edit SampleDB/example-database # manually change some settings
```

&hellip;and that's it! The operator will take care of applying the changes
as well as keeping the existing service in good shape.

## Writing your own operator {#writing-operator}

If there isn't an operator in the ecosystem that implements the behavior you
want, you can code your own. 

You also implement an operator (that is, a Controller) using any language / runtime
that can act as a [client for the Kubernetes API](/docs/reference/using-api/client-libraries/).

Following are a few libraries and tools you can use to write your own cloud native
operator.

{{% thirdparty-content %}}

---

ch object <match-objects>` instance is returned,
containing information about the match: where it starts and ends, the substring
it matched, and more.

You can learn about this by interactively experimenting with the :mod:`re`
module.

This HOWTO uses the standard Python interpreter for its examples. First, run the
Python interpreter, import the :mod:`re` module, and compile a RE::

   >>> import re
   >>> p = re.compile('[a-z]+')
   >>> p
   re.compile('[a-z]+')

Now, you can try matching various strings against the RE ``[a-z]+``.  An empty
string shouldn't match at all, since ``+`` means 'one or more repetitions'.
:meth:`~re.Pattern.match` should return ``None`` in this case, which will cause the
interpreter to print no output.  You can explicitly print the result of
:meth:`!match` to make this clear. ::

   >>> p.match("")
   >>> print(p.match(""))
   None

---

P CIDR ranges to allow as ingress sources or egress
destinations. These should be cluster-external IPs, since Pod IPs are ephemeral and unpredictable.

Cluster ingress and egress mechanisms often require rewriting the source or destination IP
of packets. In cases where this happens, it is not defined whether this happens before or
after NetworkPolicy processing, and the behavior may be different for different
combinations of network plugin, cloud provider, `Service` implementation, etc.

In the case of ingress, this means that in some cases you may be able to filter incoming
packets based on the actual original source IP, while in other cases, the "source IP" that
the NetworkPolicy acts on may be the IP of a `LoadBalancer` or of the Pod's node, etc.

For egress, this means that connections from pods to `Service` IPs that get rewritten to
cluster-external IPs may or may not be subject to `ipBlock`-based policies.

## Default policies
````

</details>

**Réponse**

```text
The information provided does not contain details about why names starting with an underscore get imported when using `from module import *`. Therefore, I cannot answer the question based on the given context.
```

**Explication du juge llama**

```text
The answer does not address the question at all, providing no information about why names starting with an underscore get imported when using `from module import *`. It simply states that the information provided does not contain details about the topic, which is not relevant to the question asked.
```


---

## p151 | relevance | llama

- **id** : p151
- **split** : test
- **config** : F-broken_retrieval
- **critère** : relevance
- **mon verdict** : PASS
- **llama** : note 1/5 (verdict au seuil par défaut : FAIL)

**Question**

```text
Why does a container with the `unless-stopped` restart policy start again after a daemon restart even when I had manually stopped it?
```

<details>
<summary>Contexte complet (tel que reçu par le générateur)</summary>

````text
/// note

Notice that even though the `item` was declared the same way as before, it is now expected to be inside of the body with a key `item`.

///

**FastAPI** will do the automatic conversion from the request, so that the parameter `item` receives its specific content and the same for `user`.

It will perform the validation of the compound data, and will document it like that for the OpenAPI schema and automatic docs.

## Singular values in body { #singular-values-in-body }

The same way there is a `Query` and `Path` to define extra data for query and path parameters, **FastAPI** provides an equivalent `Body`.

For example, extending the previous model, you could decide that you want to have another key `importance` in the same body, besides the `item` and `user`.

If you declare it as is, because it is a singular value, **FastAPI** will assume that it is a query parameter.

But you can instruct **FastAPI** to treat it as another body key using `Body`:

---

ommand.

    ```console
    $ sudo lvconvert -y \
    --zero n \
    -c 512K \
    --thinpool docker/thinpool \
    --poolmetadata docker/thinpoolmeta

    WARNING: Converting logical volume docker/thinpool and docker/thinpoolmeta to
    thin pool's data and metadata volumes with metadata wiping.
    THIS WILL DESTROY CONTENT OF LOGICAL VOLUME (filesystem etc.)
    Converted docker/thinpool to thin pool.
    ```

8.  Configure autoextension of thin pools via an `lvm` profile.

    ```console
    $ sudo vi /etc/lvm/profile/docker-thinpool.profile
    ```

9.  Specify `thin_pool_autoextend_threshold` and `thin_pool_autoextend_percent`
    values.

    `thin_pool_autoextend_threshold` is the percentage of space used before `lvm`
    attempts to autoextend the available space (100 = disabled, not recommended).

    `thin_pool_autoextend_percent` is the amount of space to add to the device
    when automatically extending (0 = disabled).

---

ers to the lower directory
as `lowerdir` and the upper directory as `upperdir`. The unified view is exposed
through its own directory called `merged`.

The `overlay2` driver natively supports up to 128 lower OverlayFS layers. This
capability provides better performance for layer-related Docker commands such
as `docker build` and `docker commit`, and consumes fewer inodes on the backing
filesystem.

### Image and container layers on-disk

After downloading a five-layer image using `docker pull ubuntu`, you can see
six directories under `/var/lib/docker/overlay2`.

> [!WARNING]
>
> Don't directly manipulate any files or directories within
> `/var/lib/docker/`. These files and directories are managed by Docker.

```console
$ ls -l /var/lib/docker/overlay2

---

t(0, x)`` inserts at the front of
   the list, and ``a.insert(len(a), x)`` is equivalent to ``a.append(x)``.


.. method:: list.remove(x)
   :noindex:

   Remove the first item from the list whose value is equal to *x*.  It raises a
   :exc:`ValueError` if there is no such item.


.. method:: list.pop([i])
   :noindex:

   Remove the item at the given position in the list, and return it.  If no index
   is specified, ``a.pop()`` removes and returns the last item in the list.
   It raises an :exc:`IndexError` if the list is empty or the index is
   outside the list range.


.. method:: list.clear()
   :noindex:

   Remove all items from the list.  Equivalent to ``del a[:]``.


.. method:: list.index(x[, start[, end]])
   :noindex:

   Return zero-based index in the list of the first item whose value is equal to *x*.
   Raises a :exc:`ValueError` if there is no such item.

---

d the nodes in
the swarm. The swarm nodes can reside on a private network that is accessible to
the proxy server, but that is not publicly accessible.

You can configure the load balancer to balance requests between every node in
the swarm even if there are no tasks scheduled on the node. For example, you
could have the following HAProxy configuration in `/etc/haproxy/haproxy.cfg`:

```bash
global
        log /dev/log    local0
        log /dev/log    local1 notice
...snip...

# Configure HAProxy to listen on port 80
frontend http_front
   bind *:80
   stats uri /haproxy?stats
   default_backend http_back

# Configure HAProxy to route requests to swarm nodes on port 8080
backend http_back
   balance roundrobin
   server node1 192.168.99.100:8080 check
   server node2 192.168.99.101:8080 check
   server node3 192.168.99.102:8080 check
```
````

</details>

**Réponse**

```text
The information provided in the context does not contain an answer to the question about the `unless-stopped` restart policy. Therefore, based on the given documentation, the relevant information is not present.
```

**Explication du juge llama**

```text
The answer explicitly states that the information provided does not contain an answer to the question, which means it does not address what was asked. Therefore, it is completely irrelevant to the question.
```

