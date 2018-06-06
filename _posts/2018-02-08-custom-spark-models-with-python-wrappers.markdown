---
layout: post
title:  "Custom Spark ML with a Python wrapper"
date:   2018-02-08 01:01:33 +0000
subtitle: Extend Spark ML functionality with your own models and provide access from python
<!-- bigimg: /img/path.jpg -->
gh-repo: raufer/custom-spark-models
gh-badge: [star, fork]
tags: [spark, scala, python]
---

Spark is a framework which tries to provides answers to many problems at once. At its core it allows for the distribution of generic workloads to a cluster. But then it provides a SQL-friendly API to work with structured data, a streaming engine to support applications with fast-data requirements and a ML library. The later is the one in which we are interested in this post: a distributed machine learning library with several models and general feature extraction, transformation and selection implementations. Supporting abstractions for composing ML pipelines or hyperparameter tunning, among others, are also provided.


Even though we get a lot out of the box from Spark ML, there will eventually be cases where you need to develop your custom transformations. Maybe the data science team you are working with as came up with some new complex features that turned out to be really valuable to the problem and now you need to implement these transformations at scale. Ideally, you will want to write them using Scala and expose a Python wrapper to facilitate their use.

For a better understanding, I recommend studying Spark's code. Start with a easy model like the `CountVectorizer` and understand what is being done. It will give you all the tools you need to build your own customizations.

We will use `Spark 2.2.1` and the `ML API` that makes use of the `DataFrame` abstraction.

The complete example can be found on this [repository][repo]. It contains the scala code plus the python wrapper implementation and boiler plate for testing in both languages.

### Custom Estimator/Transformer

Let's create a custom `Bucketizer` that will divide the range of a continuous numerical column by an input parameter `numberBins` and then, for each row, decide the appropriate bin.

Given an input column:
```
+-----+
|input|
+-----+
|  1.0|
|  5.0|
|  0.0|
|  7.0|
|  4.0|
|  8.0|
| 10.0|
+-----+
```

We expect the following output

```
+-----+---+
|input|bin|
+-----+---+
|  1.0|  0|
|  5.0|  2|
|  0.0|  0|
|  7.0|  2|
|  4.0|  1|
|  8.0|  3|
| 10.0|  4|
+-----+---+
```

In order to create a custom `Transformer` or `Estimator` we need to follow some contracts defined by Spark. Very briefly, a `Transformer` must provide a `.transform` implementation in the same way as the `Estimator` must provide one for the `.fit` method.

You need an `Estimator` every time you need to calculate something prior to the actual application of the transformation. For instance, if you need to normalize the value of the column between 0 and 1, you must necessarily first know the maximum and the minimum of that particular column. So you would create a estimator with a `.fit` method that calculates this data and then returns a Model that already has all it needs to apply the operation.

First of all declare the parameters needed by our Bucketizer:

{% gist c3b78fc786b567a4f8841cf4e900f6e7 params.scala%}

`validateAndTransformSchema` just validates the model operating conditions, like the input type of the column: `if (field.dataType!= DoubleType)`


We then declare that our `Bucketizer` will respect the `Estimator` contract, by returning a `BucketizerModel` with the `transform` method implemented. Additionally, `BucketizerParams` provides functionality to manage the parameters that we have defined above.

```
class Bucketizer(override val uid: String) extends Estimator[BucketizerModel] with BucketizerParams
```

And here is the implementation:


{% gist c3b78fc786b567a4f8841cf4e900f6e7 estimator.scala%}


The interesting part is the `fit` method that calculates the minimum and maximum values of the input column, creates a `SortedMap` with the bins boundaries and returns a `BucketizerModel` with this pre calculated data. This model, having knowledge about the boundaries, just needs to map each value to the right bin:


{% gist c3b78fc786b567a4f8841cf4e900f6e7 model.scala%}

`javaBins` is needed to map the bins data structure to a more java-friendly version. Otherwise when we ask for this structure from Python (through py4j) we cannot directly cast it to a Python `dict`

In the companion object of `BucketizerModel` we provide support for model persistence to disk.

{% gist c3b78fc786b567a4f8841cf4e900f6e7 persistence.scala%}


Spark ML has some modules that are marked as private so we need to reimplement some behaviour. In the github repository this is done in `ReadWrite.scala` and `Utils.scala`.

To create the jar:

```
sbt clean assembly
```

### Python wrapper

In case we need to provide access to our Python friends, we will need to create a wrapper on top of the Estimator.

First of all, we need to inject our custom jar to the spark context.

```python
import pyspark
from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.executor.memory", "1g")
conf.set("spark.cores.max", "2")
conf.set("spark.jars", 'spark-mllib-custom-models-assembly-0.1.jar')
conf.set("spark.app.name", "sparkTestApp")

spark = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
```

We will need to write a wrapper on top of both the `Estimator` and the `Model`. For the `Estimator` is basically just boilerplate regarding the input arguments and also specify our package name in `_classpath`.

{% gist c3b78fc786b567a4f8841cf4e900f6e7 py_estimator.py%}

`HasInputCol` and `HasOutputCol` save us the trouble of having to write:

```python
inputCol = Param(
    Params._dummy(), "inputCol", "The input column",
    typeConverter=TypeConverters.toString)

outputCol = Param(
    Params._dummy(), "outputCol", "The output column",
    typeConverter=TypeConverters.toString)
```

And here is the model:

{% gist c3b78fc786b567a4f8841cf4e900f6e7 py_model.py%}

Note that we are calling the java-friendly version to retrieve the `bins` data structure

```python
self._call_java("javaBins")
```

Additionally, we provide the qualifier name of the package where the model is implemented `com.custom.spark.feature.BucketizerModel`.

Finally, in the `read` method we are returning a `CustomJavaMLReader`. This is a custom reading behaviour that we had to reimplement in order to allow for model persistence, i.e. being able to `save/load` the model. You can check the details in the repository.

Additional support must be given to support the persistence of this model in Spark's `Pipeline` context.


<!-- ![Crepe](http://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg) -->

[repo]: https://github.com/raufer/custom-spark-models
