package com.learn.ml.classification;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class MLLinearRegression {

    @SuppressWarnings({"serial", "resource"})
    public static void main(String[] args) {
        String path = "src/main/resources/home.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load and parse the data
        JavaRDD<String> data = sc.textFile(path);
        JavaRDD<LabeledPoint> parsedData = data.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] parts = line.split(",");
                double[] v = new double[parts.length - 1];

                // normalized the data
                for (int i = 0; i < parts.length - 1; i++)
                    v[i] = Double.parseDouble(parts[i]) / 1000;

                double yvalue = Double.parseDouble(parts[parts.length - 1]) / 1000;
                return new LabeledPoint(yvalue, Vectors.dense(v));
            }
        });

        System.out.println(parsedData.count());

        // building a model
        int numIterations = 30000;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        algorithm.optimizer().setStepSize(0.00005).setNumIterations(numIterations);

        final LinearRegressionModel model = algorithm.run(JavaRDD.toRDD(parsedData));

        // Evaluate model on training examples and compute training error
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds =
                parsedData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    public Tuple2<Double, Double> call(LabeledPoint point) {
                        double prediction = model.predict(point.features());
                        double lab = point.label();
                        return new Tuple2<Double, Double>(prediction, lab);
                    }
                });

        double MSE = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {
            public Object call(Tuple2<Double, Double> pair) {
                return Math.pow(pair._1() - pair._2(), 2.0);
            }
        }).rdd()).mean();

        List<LabeledPoint> dataPoints = parsedData.collect();
        for (LabeledPoint point : dataPoints) {
            System.out.println("DataPoint: " + point.features() + " Actual Value: " + point.label() + " Expected Value:"
                    + model.predict(point.features()));
        }

        System.out.println("training Root Mean Squared Error = " + Math.sqrt(MSE));

        System.out.println(model.weights());
        System.out.println(model.intercept());


    }
}


