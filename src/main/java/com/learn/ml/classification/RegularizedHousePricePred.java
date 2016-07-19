package com.learn.ml.classification;

import java.awt.Color;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.math.plot.Plot2DPanel;

public class RegularizedHousePricePred {
    @SuppressWarnings({"serial"})
    public static void main(String[] args) {
        String path = "src/main/resources/tester.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");
        int order = 10;
        // Load and parse the data
        ParseCSVData data = new ParseCSVData(conf, path);
        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double[] v = new double[order];
                v[0] = features.toArray()[0] / 10;
                for (int i = 1; i < order; i++) {
                    v[i] = Math.pow(v[0], i + 1);
                }
                return new LabeledPoint(point.label(), Vectors.dense(v));
            }
        });


        // building a model
        RegressionModelBuilder builder = new RegressionModelBuilder(parsedData, 0.0001, 0.05);
        LinearRegressionModel model = builder.model;

        RegressionModelBuilder builder2 = new RegressionModelBuilder(parsedData, 0.0001, 0.05, 10);
        LinearRegressionModel model2 = builder2.model;

        RegressionModelBuilder builder3 = new RegressionModelBuilder(parsedData, 0.0001, 0.05, 50);
        LinearRegressionModel model3 = builder3.model;

        List<LabeledPoint> dataPoints = parsedData.collect();
        for (LabeledPoint point : dataPoints) {
            System.out.println(" Actual Value: " + point.label() + " Expected Value:" + model.predict(point.features())
                    + " DataPoint: " + point.features());
        }

        System.out.println(parsedData.count());
        System.out.println(Arrays.toString(data.getFeatures()));

        System.out.println(
                "Training Root Mean Squared Error without regularization= " + builder.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors without regularization= " + builder.getVariation());
        System.out.println("Model Equation without regularization : " + builder.getEquation());

        System.out.println(
                "Training Root Mean Squared Error with regularization  = " + builder2.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors with regularization= " + builder2.getVariation());
        System.out.println("Model Equation with regularization: " + builder2.getEquation());


        System.out.println("Training Root Mean Squared Error with high regularization param= "
                + builder3.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors with high regularization param= " + builder3.getVariation());
        System.out.println("Model Equation with high regularization param: " + builder3.getEquation());


        int len = data.getparsedData().collect().size();
        JavaRDD<LabeledPoint> features = data.getparsedData();

        double[] x = new double[len];
        double[] z1 = new double[len];
        double[] z2 = new double[len];
        double[] z3 = new double[len];
        double[] z4 = new double[len];

        // double[] weights = model.weights().toArray();
        for (int i = 0; i < dataPoints.size(); i++) {
            LabeledPoint point = dataPoints.get(i);
            double predict = model.predict(point.features());
            double predict2 = model2.predict(point.features());
            double predict3 = model3.predict(point.features());
            double[] pt = point.features().toArray();
            x[i] = pt[0];
            z2[i] = predict;
            z3[i] = predict2;
            z4[i] = predict3;
            z1[i] = point.label();
        }

        features.collect();

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot2DPanel plot = new Plot2DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, x, z1);
        plot.addScatterPlot("predict-plot", Color.green, x, z2);
        plot.addScatterPlot("predict-plot2", Color.blue, x, z3);
        plot.addScatterPlot("predict-plot3", Color.yellow, x, z4);
        plot.setAxisLabels("x", "y");

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("House plot");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);


    }
}
