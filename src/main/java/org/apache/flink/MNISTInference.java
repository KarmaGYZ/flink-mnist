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

import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink;

public class MNISTInference {

    private static final String DEFAULT_RESOURCE_NAME = "gpu";

    public static void main(String[] args) throws Exception {

        // Checking input parameters
        final ParameterTool params = ParameterTool.fromArgs(args);
        System.out.println("Usage: MNISTInference --image-file <path> [--output <path>] [--resource-name <resource_name>]");

        // set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // make parameters available in the web interface
        env.getConfig().setGlobalJobParameters(params);

        final String imageFile = params.getRequired("image-file");
        final String labelFile = params.getRequired("label-file");
        String resourceName;
        if (params.has("resource-name")) {
            resourceName = params.get("resource-name");
        } else {
            System.out.println(String.format("Executing MatrixVectorMul example with default resource name %s.\nUse --resource-name to specify resource name of GPU.", DEFAULT_RESOURCE_NAME));
            resourceName = DEFAULT_RESOURCE_NAME;
        }

        DataStream<Tuple2<Integer, Integer>> result =
                env.addSource(new MNISTReader(imageFile, labelFile))
                        .map(new MNISTClassifier(resourceName));

        // emit result
        if (params.has("output")) {
			result.addSink(
				StreamingFileSink.forRowFormat(new Path(params.get("output")),
					new SimpleStringEncoder<Tuple2<Integer, Integer>>()).build());
        } else {
            System.out.println("Printing result to stdout. Use --output to specify output path.");
            result.print();
        }
        // execute program
        env.execute("MNIST Inference");
    }
}
