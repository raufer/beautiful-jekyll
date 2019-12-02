---
layout: post
title:  "The Open approach to Operationalize Machine Learning"
date:   2019-08-02 09:01:33 +0000
subtitle: Constructing an open-source ecosystem to govern the complexity of Machine Learning services
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

![platform]({{ "/img/ops-ml/platform-components-2.png" | absolute_url}})


### ML Primitives

Lets briefly describe the responsabilities of each modular component. We'll also look to the current technology landscape and suggest tools that can solve the challenges posed by each layer.

#### Model Training

Training machine learning models can be a highly time-consuming task. It can also be demanding from a resources allocation prespective, since bigger datasets and sophisticated models might require considerable amounts of processing power, which sometimes needs to come in specialized form, e.g. GPUs, TPU, etc.

As *Auto Machine Learning* technology improves, the process by which we select a  model, along with its tunned hyperparameters, is becoming more standardized. This wide search space is being explored by various methods like Bayesion Optimization or Genetic Programming. However, these algorithms can take a long time to finish their search.

Most of the times they aren't as simple as fitting one model on the dataset; they are considering multiple machine learning algorithms (random forests, linear models, SVMs, etc.) in a pipeline with multiple preprocessing steps (missing value imputation, scaling, PCA, feature selection, etc.), the hyperparameters for all of the models and preprocessing steps, as well as multiple ways to ensemble or stack the algorithms within the pipeline.

As an example consider a Genetic Programming library to evolve a ML pipeline that considers various preprocessing steps and multiple machine learning algorithms; Assume a configuration of `100 generations` with `100 population size`; This would result in 10,000 model configurations to evaluate with 10-fold cross-validation, which means that roughly 100,000 models are fit and evaluated on the training data. This is a time-consuming procedure, even for simpler models like decision trees.

We need an orchestration tool that allows us to orchestrate machine learning training pipeline at scale. [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/) is such a tool, belonging to Google's [Kubeflow](https://www.kubeflow.org/) ML ecosystem, a project dedicated to making deployments of machine learning (ML) workflows on Kubernetes.

Kubeflow Pipelines is Kubernetes-native workflow orchestrator that builds on top of [Argo Workflows](https://argoproj.github.io/argo/); it isn't a typical orchestrator in the sense that it knows the structure of a typical ML workflow and so provides features that leverage that knowledge. It has a built-in `Experiment` concept that can be used to run a workflow with a set of parameters so that we can then track the associated performance metrics. It also allows a workflow to output
different artifacts that are used to assess the model performance/behaviour. 

Here's an illustration of a workflow:

![platform]({{ "/img/ops-ml/kubeflow-1.png" | absolute_url}})

Every step runs in its own container, allowing us to: isolate software dependencies; optimize required resources for each step; and easily parallelize parts of the workflow.

#### Model Serving

Once a training process results in a satisfactory model, the later should be made available at a central repository to facilitate its integration into a service.

We need a framework agnostic tool to rapidly deploy pipelines for model inference at scale. Note that, in the most general case, the transformations that occur at *training time* are not necessarily the same that we can afford to perform at *inference time*. There is some differentiation between transformation *training* and *testing* data; this is because training sets are typically enhanced by processing the full dataset all at once to extract meaningfull information, while at *inference time* the data must be processed row by row (assuming a real-time system), leading to some information
loss when compared to the *training phase*. 

Moreover, we need a tool that provides us to create flexible inference pipelines that can go beyhond the traditional pattern of receiving the data and calling the model's API. Here's an illustration of a possible inference pipeline:

![platform]({{ "/img/ops-ml/seldon-2.png" | absolute_url }})

In this example the service receives some input data; transforms it to extract the relevant features; enriches the data with a query to some database to provide extra context; and then asks a prediction from multiple models.o

Being able to create arbitrarily complex inference graphs creates the opportunity to generalize common behaviour like creating generic steps for outlier detection or performing multi-armed bandits tests.

[Seldon](https://github.com/SeldonIO/seldon-core) is a model serving framework that supports model deployment on a kubernetes cluster. It allows for the creating of inference graphs, where each step runs as a different container.

Every graph has a model orchestration pod at the top with a standardized API contract for both REST and gRPC. When it receives a request (or a batch of requests) it then triggers the execution of the graph. The of the wiring between the steps is handled by Seldon.

We can also leverage Kubernete's inherent scalability capability to serve models at scale. Here's the relevant part of a Seldon's graph definition:


```yaml
hpaSpec:
  minReplicas: 1
  maxReplicas: 4
    metrics:
      - type: Resource
        resource:
          name: cpu
          targetAverageUtilization: 0.80
```

This in turn will create Horizontal Pod Autoscaler (HPA) rules in Kubernetes that monitor the service metrics. Once the metric threshold is not respected, a service replica is created to cope with the increased traffic.

```bash
❯ kubectl get hpa --all-namespaces

NAMESPACE      NAME                          REFERENCE                                TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
jx-staging     manticore-manticore-a0b1623   Deployment/manticore-manticore-a0b1623   39%/80%   1         4         1          28d
```

To expose the models to consumers outside of the cluster Seldon integrates with [istio](https://istio.io/). Everytime that the Seldon Kubernetes Operator detects a request for a Seldon deployment, it injects the necessary instructions to wire the service to `istio`. `L7 Routing` capability is also available via the integration with `istio`.

Each deployment exposes two default endpoints: `predict` and `feedback`:

![platform]({{ "/img/ops-ml/seldon-4.png" | absolute_url }})

The `predict` is the one used to trigger the execution of the graph to get an inference. We can use the `feedback` endpoint to capture responses, when these are made available. The later is important for:

* Help deciding which model should serve more traffic when running inference graphs with A/B or multi-armed bandits;
* Models that support online training;
* Capturing new labelled data to be used in retraining jobs;

#### Model Observability

Once a model is live, we do need to be able to monitor technical metrics like the number of requests/sec; the response latency; 

#### Governance

#### Data Exploration




==>>>

The pipeline to transform data for training is different from the pipeline that transforms data at inference time

Kubernetes is able to provide answers to the infrastcuture

The platform wont be ready without support for pregressive delivery.

The treat code and data changes seamlessly in the same way, we cannot explicit consume data from within the ML workflows. A declarative approach is needed, where we declare the intended version of the data.



