package com.learn.ml.classification;

import java.util.Arrays;

import org.apache.spark.SparkConf;

public class Algo {

    public static void main(String[] args) {
        String filepath = "src/main/resources/3D_spatial_network.csv"; // Should be some file on
                                                                       // your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");

        ParseCSVData data = new ParseCSVData(conf, filepath);
        LRModelBuilder builder = new LRModelBuilder(data.getparsedData(), 0.0001, 0.000001);

        System.out.println("Fearures: " + Arrays.toString(data.getFeatures()));
        System.out.println("LMSE: " + builder.getLeastMeanSquareError());
        System.out.println("Coefficients: " + builder.model.weights());
        System.out.println("Intercept:" + builder.model.intercept());
    }

}
