---
layout: post
title:  "Operationalizing Machine Learning"
date:   2019-08-02 09:01:33 +0000
subtitle: Constructing an ecosystem to handle the complexity of Machine Learning services
<!-- bigimg: /img/path.jpg -->
gh-repo: raufer/flow-writer
gh-badge: [star, fork]
tags: [machine-learning, operations, machine learning platform, machine learning operationalization]
---

This blog post discusses the operationalization of machine learning systems. From an engineering prespective, these systems pose enormous challenges due to the high variability of requirements that different use cases have. As the machine-leraning maturity of organizations evolves, the complexity required to manage these systems grows exponentially; The proliferation of ML frameworks makes it hard to abstract layers of common functionality and, consequently, automating the whole process becomes even harder. 

As regulators move towards giving consumers the "right to explainability" and the "right to model retraining", compliance requirements bring a new dimension of complexity to the processes. 

Ideally, organizations should have standardized processes by which they can deliver ML services. This requires a ML ecosystem/platform that considers the following (non exhaustive) properties:

* **Scalability** - Working with samples and small datasets is sometimes enough the initial research phase. However, teams eventually require to have access to distributed data processing and model training. Furthermore, we might need to data pipelines and models at scale. The underlying platform-infrastructure is responsible to provide this capability.

* **Framework Agnostic** - Teams should have the flexibility to use any required tool; this is particularly relevant as frameworks become more specialized. This also minimzes code-rewrites between research and operations which is highly error prone.

* **Modular** - The platform should provide primitive components that can be used by the engineering teams to deliver ML on a standardized way. We must identify commonalities for modularization. 

*  **Governance** - From a technical point of view, we need enough evidence to prove that the delivery process produces artifacts that are reproducible, traceable, and verifiable.
Further, there are complex ethical, legal, conduct and reputational issues associated with the use of personal
data. For example, are data being used unfairly to exclude individuals or groups, or to promote unjustifiably
privileged access for others? Controls are needed to ensure data is treated fairly.

We are going to propose a technology stack for a ML-platform solution that is able to describe processes respecting the desired properties. The following illustratres the different components involved:

![platform]({{ "/img/ops-ml/platform-components.png" | absolute_url | width=50}})


...

### ML Primitives

Lets briefly describe the responsabilities of each modular component. We'll also look to the current technology landscape and suggest tools that can solve the challenges posed by each layer.

#### Model Training

#### Model Serving

Once a training process results in a satisfactory model, the later should be made available at a central repository to facilitate its integration into a service.

We need a framework agnostic tool to rapidly deploy pipelines for model inference at scale. Note that, in the most general case, the transformations that occur at *training time* are not necessarily the same that we can afford to perform at *inference time*. There is some differentiation between transformation *training* and *testing* data; this is because training sets are typically enhanced by processing the full dataset all at once to extract meaningfull information, while at *inference time* the data must be processed row by row (assuming a real-time system), leading to some information
loss when compared to the *training phase*. 

Moreover, we need a tool that provides us to create flexible inference pipelines that can go beyhond the traditional pattern of receiving the data and calling the model's API. Here's an illustration of a possible inference pipeline:

![platform]({{ "/img/ops-ml/seldon-1.png" | absolute_url }})

In this example the service receives some input data; transforms it to extract the relevant features; enriches the data with a query to some database to provide extra context; and then asks a prediction from multiple models.o

Being able to create arbitrarily complex inference graphs creates the opportunity to generalize common behaviour like creating generic steps for outlier detection or performing multi-armed bandits tests.

[Seldon](https://github.com/SeldonIO/seldon-core) is a model serving framework that supports model deployment on a kubernetes cluster. It allows for the creating of inference graphs, where each step runs as a different container.

Every graph has a model orchestration pod at the top with a standardized API contract for both REST and gRPC. When it receives a request (or a batch of requests) it then triggers the execution of the graph. The of the wiring between the steps is handled by Seldon.



#### Model Observability

#### Governance

#### Data Exploration




==>>>

The pipeline to transform data for training is different from the pipeline that transforms data at inference time

Kubernetes is able to provide answers to the infrastcuture

The platform wont be ready without support for pregressive delivery.

The treat code and data changes seamlessly in the same way, we cannot explicit consume data from within the ML workflows. A declarative approach is needed, where we declare the intended version of the data.



