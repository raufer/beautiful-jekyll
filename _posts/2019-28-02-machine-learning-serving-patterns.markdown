---
layout: post
title:  "Model Serving Patterns"
date:   2019-09-01 13:01:33 +0000
subtitle: Overview of the typical patterns of consumption of machine learning models
<!-- bigimg: /img/path.jpg -->
tags: [model serving, model serving patterns, ml serving, opsml, ml ops, machine learning deployments, machine-learning, operations, machine learning platform, machine learning operationalization]
---
In a [previous post](https://raufer.github.io/2019/08/02/operationalizing-machine-learning/) we have discussed in some detail the design of a machine learning platform covering model training, serving, governance and observability.

We are now going to look with more detail to **model serving**:

* the different model consumption patterns;
* technical solutions to each one;
* comments on the additional complexity required by the solution to accommodate all the different patterns;


### The Patterns

1. **Online Stateless**

The consumption is done on an event basis, hence the model needs to exposed behind a service, e.g. `REST`, `gRPC`. The consumer has the ownership of sending all of the necessary data for the inference operation as dictated by the API contract. Useful to obtain inferences for a single input or a batch of them. Acceptable response times are typically on the subsecond range. Each request is independent and therefore no state needs to be mantained.

![online-statless]({{ "/img/ml-serving/online-stateless2.png" | absolute_url}})

For this consumption pattern, the service exposing the model needs to be operational on a continuous basis since the service does not know in advance when consumers are going to request an inference.

In terms of scalability we are primarily interested in responding to and *increased number of individual consumers* and not to *increases in the volume of data*  being passed to the service; in this pattern we assume that the consumer is interested in obtaining inferences for one/small-batch of examples. More suitable patterns should be used if a particular consumer needs to obtain inferences for arbitrarily large datasets.

2. **Online Stateful**

This consumption pattern is required for more complex scenarios where the the consumer might require some level of interactivity with the model service.

![online-stateful]({{ "/img/ml-serving/online-stateful2.png" | absolute_url}})

E.g.:

- Consumer `C1` makes a call to a service `S` asking for a recommendation;
- Service `S` responds back with recommendation `R1`;
- Consumer `C1` is not satisfied and asks the service `S` to try again;
- Service `S` answers back with a second recommendation `R2`;

This scenario requires some form of state persistance across calls for the different consumers.


3. Offline in Batches.

For scenarios where it is requried to perform model inference at scale, an offline batch pipeline is the best option. Obviously because we can make use of tansient infrastructure to serve the inference job. This pattern also differs from the `online` patterns with regards to the location of the data; for the online patterns the consumer has the responsability to send the data; whilst for offline inference we expect the data to be located at a known storage.

![offline-batch]({{ "/img/ml-serving/offline-batch2.png" | absolute_url}})

For smaller datasets, we could in theory have a consumer sending out a stream of batches to the online service but this extra logic would need to be implemented and mantained at the consumer side. Data processing frameworks like `Apache Spark` can easily distribute and collect the results.

The resulting inferences are then typically saved back to a storage system for later consumption by downstream applications, e.g. dashboards.


### The Solution

![solution]({{ "/img/ml-serving/solution2.png" | absolute_url}})

### Final Comments




