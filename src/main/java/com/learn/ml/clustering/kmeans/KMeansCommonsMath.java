package com.learn.ml.clustering.kmeans;


import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import com.learn.ml.clustering.Clusterer;
import com.learn.ml.clustering.Vector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.MultiKMeansPlusPlusClusterer;

/**
 * Class KMeansCommonsMath.
 *
 */
public class KMeansCommonsMath implements Clusterer {

    private final VectorDataProvider<?> provider;

    public KMeansCommonsMath(VectorDataProvider provider) {
        this.provider = provider;
    }

    @Override
    public <T extends Vector> List<CentroidCluster<T>> getClusters(List<T> vectors, int numberOfClusters) {
        KMeansPlusPlusClusterer<T> clusterer = new KMeansPlusPlusClusterer<T>(numberOfClusters);

        MultiKMeansPlusPlusClusterer<T> multiCluster = new MultiKMeansPlusPlusClusterer<>(clusterer, 100000);
        return multiCluster.cluster(vectors);
    }

    public static void main(String[] args) throws Exception {
        VectorDataProvider<?> provider = new EmployeeDataProvider("newData.csv");

        KMeansCommonsMath kmeans = new KMeansCommonsMath(provider);



        List<? extends CentroidCluster<?>> clusters = kmeans.getClusters(4);
        System.out.println(isFairClustering(clusters));
        Map<CentroidCluster, Double> maxDistanceMap = getMaxDistanceMap(clusters);
        final int[] index = {1};
        clusters.forEach(new Consumer<CentroidCluster<?>>() {
            @Override
            public void accept(CentroidCluster<?> centroidCluster) {
                System.out.println("Cluster:" + index[0]++ + " Size: " + centroidCluster.getPoints().size());
                Clusterable center = centroidCluster.getCenter();
                Vector centroid = new Vector(center.getPoint());

                double[] pts = centroid.getPoint();
                for (int i = 0; i < centroid.getPoint().length; i++) {
                    double value = (double) Math.round(pts[i] * 10000) / 100;
                    pts[i] = value;
                }
                System.out.println(String.format("%30s : %2.20f : %s ", "Maximum", maxDistanceMap.get(centroidCluster),
                        Arrays.toString(pts) + " Dimension : " + centroid.getPoint().length));

                List<EmployeeVector> points = (List<EmployeeVector>) centroidCluster.getPoints();
                points.forEach(employeeVector -> {
                    System.out.println(String.format("%30s : %2.20f : %s", employeeVector.getName(),
                            employeeVector.distanceTo(centroid), Arrays.toString(employeeVector.getPoint())));
                });
                System.out.println("\n\n");

            }
        });
        // System.out.println(clusters);
    }

    private static boolean isFairClustering(List<? extends CentroidCluster<?>> clusters) {
        Map<CentroidCluster, Double> maxDistanceMap = getMaxDistanceMap(clusters);
        for (int i = 0; i < clusters.size(); i++) {
            CentroidCluster<?> c1 = clusters.get(i);
            for (int j = 0; j < i; j++) {
                System.out.print(i + " : " + j);
                CentroidCluster<?> c2 = clusters.get(j);
                Vector v1 = new Vector(c1.getCenter().getPoint());
                Vector v2 = new Vector(c2.getCenter().getPoint());
                double c_distance = v1.distanceTo(v2);
                if (c_distance - (maxDistanceMap.get(c1) + maxDistanceMap.get(c2)) < 0) {
                    System.out.print(" breaking");
                }
                System.out.print("\n");
            }
        }
        return false;
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

}
