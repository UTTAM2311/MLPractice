package com.learn.ml.clustering;

import java.util.List;

import org.apache.commons.math3.ml.clustering.CentroidCluster;

/**
 * Class ${name}.
 * 
 */
public interface Clusterer {

    <T extends Vector>List<CentroidCluster<T>> getClusters(List<T> vectors, int numberOfCluster);
}
