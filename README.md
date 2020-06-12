# flink-mnist
Run MNIST inference in Apache Flink

# Usage

1. Package the project.

```
mvn clean package # you could get MNISTInference.jar in target/ folder
```

2. Download the native libraries of JCuda and JCublas to lib/ folder of your Flink distribution

```
./add-jcuda-dependency.sh <your_flink_dist>
```

3. Prepare image data. You could download the [training data](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
or [test data](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz). Then, decompress it by `gzip xxx.gz`.

4. Configure to enable GPU plugin, please refer to user document.

5. Start the Flink cluster and run the MNIST inference job.

```
cd $FLINK_HOME
bin/start-cluster.sh
bin/flink run path/to/MNISTInference.jar --image-file path/to/imagefile
```