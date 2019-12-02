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


## ML Primitives

Lets briefly describe the responsabilities of each modular component. We'll also look to the current technology landscape and suggest tools that can solve the challenges posed by each layer.

### Model Training

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

### Model Serving

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

<br/><br/>

### Model Observability

Once a model is live, we do need to be able to monitor technical metrics like the number of requests/sec or the response latency.

However, data science teams should have observe more functional metrics like the real time performance of the model, the bias in the response or the distribution of the input data.This would provide useful insights to the teams to detect possible drifts in the target signal, or on how to make the model fairer.

Seldon provides an awesome integration with [Grafana](https://grafana.com/dashboards) and [Prometheus](https://prometheus.io/). For each Seldon Deployment, the Seldon's Service Orchestrator exposes a set of default Prometheus metrics which are then displayed in a central Grafana dashboard.

Here's an illutratrion of what this looks like:

![platform]({{ "/img/ops-ml/dashboard-1.png" | absolute_url }})


Custom metrics can be easily exposed and integrated with these dashboards. Seldon takes care of all of wiring and communication among the services. At the dasboard level, we are able select different deployments, versions or even particular containers that form a inference graph.

### Governance

A model's behaviour is affected not only by its code and configuration but also by data consumed at training time. The following is a non exhaustive list of what is needed to achieve governance when deploying ML services:

* Every Machine Learning Asset should be versioned. This includes code, data, configuration, models, inference graphs etc.
* There should be visibility over the different experiments performed and associated metrics;
* Every model should be reproducible;
* We should be able to verify the results of a Machine Learning model to ensure we are only integrating models with acceptable performance;
* Rollback mechanisms should be in place;

A ML model is not much different from data: a set of learned weights along with some metadata. Any artifact storage solution should suffice to be used as a model repository. Nexus or a object storage like MongoDB can be used as a model repository.

Experiments tracking can be done at the orchestrator level, e.g. Kubeflow Pipelines. If a more specialized service is need, a [ML Flow Tracking](https://www.mlflow.org/docs/latest/tracking.html) service can be used. The later is speciality relevant if we also need to track experiments made data scientist at *exploration phase*, i.e. when they ae exploring the data and testing out ideas. MLFlow provides a simple `python`/`R` library that can integrated in a data scientist exploration
environment to track performance metrics and parameters used.

e.g.:

```python
import mlflow

...

with mlflow.start_run():

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Set tracking_URI first and then reset it back to not specifying port
    # Note, we had specified this in an earlier cell
    mlflow.set_tracking_uri(mlflow_tracking_URI)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
```

With such a service we can centralize the information related to every ML experiment:

* Experiments performed by the CICD infrastructure as a response to a data/code change; 
* Experiments performed by data scientists while they are exploring new ideas;

The `MLFlow` artifact store can also be used as a model repository.

Treating a data change in the same way as a code change requires tooling to support some kind of formal *data versioning*.
[Pachyderm](https://www.pachyderm.com/) is Kubernetes framework that can help us with the engineering side of ML workflows. It can used as *git for data*.

This also means that we our training workflows cannot explicitly consume the data when an execution is triggered. The consumption must be declarative and done via `git`.
By doing so, the CICD pipeline can respond in the same way to changes in both code and data.

Pachyderm provides an intuitive solution for data versioning. The following snippets illustrate the typical workflow when working with Pachyderm (via the `patchctl` CLI).

Create a data repository:

```bash
pachctl create repo product-recommendation
```

Commit data to master branch:

```bash
pachctl put file -r product-recommendation@master:/data -f data/
```

We can then list all of the different versions of the `master` branch:


```bash
❯ pachctl list commit product-recommendation

REPO                   BRANCH COMMIT                           PARENT STARTED    DURATION           SIZE
product-recommendation master 70c9c944ed5f4d7286d5f16461d07c91 <none> 3 days ago Less than a second 52.4MiB
```

Note that `master` is just a pointer to the latest commit hash. To illustrate this consider the following example:

```bash
❯ pachctl list commit product-recommendation
REPO                   BRANCH COMMIT                           PARENT STARTED      DURATION           SIZE
product-recommendation master e37161e974e8482bbccc89e01b414f7c <none> 26 hours ago Less than a second 3.768KiB
product-recommendation master c3f0233361fd43f39f532ec6859001c8 <none> 26 hours ago 1 second           3.768KiB
```

The folllowing commands are equivalent:

```bash
pachctl get file product-recommendation@master:/data/ --recursive --output .
pachctl get file product-recommendation@e37161e974e8482bbccc89e01b414f7c:/data/ --recursive --output .
```

Every change must have associated a particular version of the code and data. This is why the data cannot be consumed directly and must be done via `git`.

A simple way would be to have the data version as a configuration in the repository:

`/configuration/data.json`

```json
{
  "train": {
      "commit": "befcd45a69704b3094f10a031ffaf398"
  }
}
```

The CICD pipeline would be triggered in two ways:

1. Code Change

The team changes the logic of the model training process, e.g. new features, more complex model, etc.

2. Data Changes

Every time data changes, a pipeline must be triggered that will version it in Pachyderm and commit a change to the relevant repositories by changing `train.commit` in `/configuration/data.json`; this in turn will trigger an execution of the training process.


### Data Exploration

Teams require the ability to create data exploration environments, from which they can access the data to research possible solutions to the problems they are working on.
Special purpose hardware like GPUs should be made available on demand when needed. 

Different flavours of these environments, with different technological stacks, can provide development velocity and flexibility to different teams.

[JupyterHub](https://jupyterhub.readthedocs.io/) or [Anaconda](https://www.anaconda.com/) are well known tools that provide user-sessions on top of Kubernetes.


## Workflow Diagram




==>>>

The pipeline to transform data for training is different from the pipeline that transforms data at inference time

Kubernetes is able to provide answers to the infrastcuture

The platform wont be ready without support for pregressive delivery.

The treat code and data changes seamlessly in the same way, we cannot explicit consume data from within the ML workflows. A declarative approach is needed, where we declare the intended version of the data.



