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

import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.source.RichSourceFunction;
import org.apache.flink.util.Preconditions;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

import static org.apache.flink.MNISTModel.DIMENSIONS;

/**
 * MNIST data reader.
 */
class MNISTReader extends RichSourceFunction<List<Float>> {

    private final String imageFilePath;

    private transient volatile boolean running;

    MNISTReader(String imageFilePath) {
        this.imageFilePath = imageFilePath;
    }

    @Override
    public void open(Configuration parameters) {
        running = true;
    }

    @Override
    public void run(SourceContext<List<Float>> ctx) throws Exception {
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFilePath)));
        // read magic number
        dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        // read row and col dimensions.
        Preconditions.checkState(dataInputStream.readInt() == 28);
        Preconditions.checkState(dataInputStream.readInt() == 28);

        int count = 0;
        while (running && count < numberOfItems) {
            List<Float> data = new ArrayList<>(DIMENSIONS.f0);
            for (int i = 0; i < DIMENSIONS.f0; ++i) {
                data.add((float) dataInputStream.readUnsignedByte());
            }
            ctx.collect(data);
            count += 1;
        }
        dataInputStream.close();
    }

    @Override
    public void cancel() {
        running = false;
    }
}
