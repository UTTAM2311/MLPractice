package com.learn.ml.clustering.kmeans;/**
                                        * Class ${name}.
                                        *
                                        * @author nagarajan
                                        * @version 1.0
                                        * @since
                                        * 
                                        *        <pre>
                                        * 1/6/16 5:15 PM
                                        *        </pre>
                                        */


import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.learn.ml.clustering.Clusterer;
import com.learn.ml.clustering.Vector;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;

/**
 * Class KMeansCustomImpl.
 */
public class KMeansCustomImpl implements Clusterer {
    private final VectorDataProvider<?> provider;

    public KMeansCustomImpl(VectorDataProvider provider) {
        this.provider = provider;
    }

    @Override
    public <T extends Vector> List<CentroidCluster<T>> getClusters(List<T> vectors, int numberOfClusters) {
        return cluster(vectors, numberOfClusters);
    }

    private <T extends Vector> List<CentroidCluster<T>> cluster(List<T> vectors, int numberOfClusters) {
        if (vectors == null || vectors.isEmpty() || vectors.size() == numberOfClusters) {
            throw new RuntimeException("Invalid data for clustering");
        }

        Vector originVector = getOriginVector(vectors.get(0));
        List<DistanceVector> collect =
                vectors.stream().map(v -> new DistanceVector(v, originVector)).sorted().collect(Collectors.toList());
        return null;
    }

    private <T extends Vector> Vector getOriginVector(T t) {
        double[] origin = new double[t.length()];
        for (int i = 0; i < t.length(); i++) {
            origin[i] = 0;
        }
        return new Vector(origin);
    }

    public static void main(String[] args) throws Exception {
        VectorDataProvider<?> provider = new EmployeeDataProvider("data.csv");
        KMeansCustomImpl kmeans = new KMeansCustomImpl(provider);
        List<? extends CentroidCluster<?>> clusters = kmeans.getClusters(3);
        System.out.println(clusters);
    }

    private static Map<CentroidCluster, Double> getMaxDistanceMap(List<? extends CentroidCluster<?>> clusters) {
        Map<CentroidCluster, Double> maxDistanceMap = new HashMap<>(clusters.size());
        for (CentroidCluster<?> cluster : clusters) {
            maxDistanceMap.put(cluster, getMaxDistance(cluster));
        }
        return maxDistanceMap;
    }

    private static double getMaxDistance(CentroidCluster<?> centroidCluster) {
        Vector centroid = new Vector(centroidCluster.getCenter().getPoint());
        double max = Double.MIN_VALUE;
        for (Clusterable c : centroidCluster.getPoints()) {
            Vector v = (Vector) c;
            max = Math.max(max, v.distanceTo(centroid));
        }
        return max;
    }

    private List<? extends CentroidCluster<?>> getClusters(int i) throws Exception {
        return getClusters(provider.getVectorData(), i);
    }

    private class DistanceVector implements Comparable<DistanceVector> {
        private final Vector origin;
        private final Vector vector;
        private final double distance;

        DistanceVector(Vector originVector, Vector t) {
            this.origin = originVector;
            this.vector = t;
            this.distance = origin.distanceTo(vector);
        }

        public double getDistance() {
            return distance;
        }

        public Vector getVector() {
            return vector;
        }

        public Vector getOrigin() {
            return origin;
        }


        @Override
        public String toString() {
            return new ToStringBuilder(this).append("origin", origin).append("vector", vector)
                    .append("distance", distance).toString();
        }

        @Override
        public int compareTo(DistanceVector o) {
            return Double.compare(distance, o.getDistance());
        }
    }
}
