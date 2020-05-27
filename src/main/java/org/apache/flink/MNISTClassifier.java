/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink;

import org.apache.flink.api.common.externalresource.ExternalResourceInfo;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Preconditions;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.apache.flink.MNISTModel.DIMENSIONS;

/**
 * MNIST classifier.
 */
class MNISTClassifier extends RichMapFunction<List<Float>, Integer> {

    private final String resourceName;
    private Pointer matrixPointer;

    MNISTClassifier(String resourceName) {
        this.resourceName = resourceName;
    }

    @Override
    public void open(Configuration parameters) {
        // When multiple instances of this class and JCuda exist in different class loaders, then we will get UnsatisfiedLinkError.
        // To avoid that, we need to override the java.io.tmpdir, where the JCuda store its native library, with a random path.
        final String originTempDir = System.getProperty("java.io.tmpdir");
        final String newTempDir = originTempDir + "/jcuda-" + UUID.randomUUID();
        System.setProperty("java.io.tmpdir", newTempDir);

        final Set<ExternalResourceInfo> externalResourceInfos = getRuntimeContext().getExternalResourceInfos(resourceName);
        Preconditions.checkState(!externalResourceInfos.isEmpty(), "The MatrixVectorMul needs at least one GPU device while finding 0 GPU.");
        final Optional<String> firstIndexOptional = externalResourceInfos.iterator().next().getProperty("index");
        Preconditions.checkState(firstIndexOptional.isPresent());

        matrixPointer = new Pointer();

        // Set the CUDA device
        JCuda.cudaSetDevice(Integer.parseInt(firstIndexOptional.get()));

        // Initialize JCublas
        JCublas.cublasInit();

        // Allocate device memory for the matrix
        JCublas.cublasAlloc(DIMENSIONS.f0 * DIMENSIONS.f1, Sizeof.FLOAT, matrixPointer);
        JCublas.cublasSetVector(DIMENSIONS.f0 * DIMENSIONS.f1, Sizeof.FLOAT, Pointer.to(MNISTModel.MODEL), 1, matrixPointer, 1);

        // Change the java.io.tmpdir back to its original value.
        System.setProperty("java.io.tmpdir", originTempDir);
    }

    @Override
    public Integer map(List<Float> value) {
        final float[] input = new float[DIMENSIONS.f0];
        final float[] output = new float[DIMENSIONS.f1];
        final Pointer inputPointer = new Pointer();
        final Pointer outputPointer = new Pointer();

        // Fill the input and output matrix
        for (int i = 0; i < DIMENSIONS.f0; i++) {
            input[i] = value.get(i);
        }
        for (int i = 0; i < DIMENSIONS.f1; i++) {
            output[i] = 0;
        }

        // Allocate device memory for the matrices
        JCublas.cublasAlloc(DIMENSIONS.f0, Sizeof.FLOAT, inputPointer);
        JCublas.cublasAlloc(DIMENSIONS.f1, Sizeof.FLOAT, outputPointer);

        // Initialize the device matrices
        JCublas.cublasSetVector(DIMENSIONS.f0, Sizeof.FLOAT, Pointer.to(input), 1, inputPointer, 1);
        JCublas.cublasSetVector(DIMENSIONS.f1, Sizeof.FLOAT, Pointer.to(output), 1, outputPointer, 1);

        // Performs operation using JCublas
        JCublas.cublasSgemv('n', DIMENSIONS.f1, DIMENSIONS.f0, 1.0f,
                matrixPointer, DIMENSIONS.f1, inputPointer, 1, 0.0f, outputPointer, 1);

        // Read the result back
        JCublas.cublasGetVector(DIMENSIONS.f1, Sizeof.FLOAT, outputPointer, 1, Pointer.to(output), 1);

        // Memory clean up
        JCublas.cublasFree(inputPointer);
        JCublas.cublasFree(outputPointer);

        int result = 0;
        for (int i = 0; i < DIMENSIONS.f1; ++i) {
            result = output[i] > output[result] ? i : result;
        }

        return result;
    }

    @Override
    public void close() {
        // Memory clean up
        JCublas.cublasFree(matrixPointer);

        // Shutdown cublas
        JCublas.cublasShutdown();
    }
}
