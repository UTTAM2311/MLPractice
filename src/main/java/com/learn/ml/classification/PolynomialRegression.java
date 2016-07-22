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

public class PolynomialRegression {
    public static void main(String[] args) {
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        int order = 4;
        drawPlots(path, order, 10000, 0.03, 0);
    }

    @SuppressWarnings("serial")
    public static void drawPlots(String path, int order, int iter, double stepSize, double regParam) {
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");

        // Load and parse the data
        ParseCSVData data = new ParseCSVData(conf, path);
        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double[] v = new double[order];
                v[0] = features.toArray()[0] / 10000;
                for (int i = 1; i < order; i++) {
                    v[i] = Math.pow(v[0], i + 1);
                }
                return new LabeledPoint(point.label() / 10000, Vectors.dense(v));
            }
        });

        // building a model
        RegressionModelBuilder builder = new RegressionModelBuilder(parsedData, iter, stepSize, regParam);
        LinearRegressionModel model = builder.model;

        List<LabeledPoint> dataPoints = parsedData.collect();
        for (LabeledPoint point : dataPoints) {
            System.out.println(" Actual Value: " + point.label() + " Expected Value:" + model.predict(point.features())
                    + " DataPoint: " + point.features());
        }

        System.out.println(parsedData.count());
        System.out.println(Arrays.toString(data.getFeatures()));

        System.out.println("Training Root Mean Squared Error  = " + builder.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors = " + builder.getVariation());
        System.out.println("Model Equation : " + builder.getEquation());
        int len = data.getparsedData().collect().size();
        JavaRDD<LabeledPoint> features = data.getparsedData();

        double[] x = new double[len];
        double[] z1 = new double[len];
        double[] z2 = new double[len];

        // double[] weights = model.weights().toArray();
        for (int i = 0; i < dataPoints.size(); i++) {
            LabeledPoint point = dataPoints.get(i);
            double predict = model.predict(point.features());
            double[] pt = point.features().toArray();
            x[i] = pt[0];
            z2[i] = predict;
            z1[i] = point.label();
        }

        features.collect();

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot2DPanel plot = new Plot2DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, x, z1);
        plot.addScatterPlot("predict-plot", Color.green, x, z2);
        plot.setAxisLabels("x", "y");

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("House plot");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

}
