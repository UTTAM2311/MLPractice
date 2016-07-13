package com.learn.ml.classification;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;

public class HousePricePrediction {

    @SuppressWarnings({"serial"})
    public static void main(String[] args) {
        String path = "src/main/resources/home.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");

        // Load and parse the data
        ParseCSVData data = new ParseCSVData(conf, path);
        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double[] v = new double[features.size()];
                for (int i = 0; i < v.length; i++) {
                    v[i] = features.toArray()[i] / 1000;
                }
                return new LabeledPoint(point.label() / 1000, Vectors.dense(v));
            }
        });

        System.out.println(parsedData.count());

        // building a model
        RegressionModelBuilder builder = new RegressionModelBuilder(parsedData, 0.00001, 0.0001);
        LinearRegressionModel model = builder.model;
        List<LabeledPoint> dataPoints = parsedData.collect();


        for (LabeledPoint point : dataPoints) {
            System.out.println(" Actual Value: " + point.label() * 1000 + " Expected Value:"
                    + model.predict(point.features()) * 1000 + " DataPoint: " + point.features());
        }

        System.out.println("Training Root Mean Squared Error = " + builder.getLeastMeanSquareError() * 1000);
        System.out.println("Mean Variation in the errors = " + builder.getVariation() * 1000);
        System.out.println("Model Equation: " + builder.getEquation());

    }
}


