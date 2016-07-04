package com.learn.ml.clustering.kmeans;

import java.util.List;
import com.learn.ml.clustering.Vector;

/**
 * Class VectorDataProvder.
 */
public interface VectorDataProvider<T extends Vector> {

    List<T> getVectorData() throws Exception;
}
