package com.learn.ml.classification;

import java.awt.Color;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.math.plot.Plot3DPanel;

public class RegressionAnalyzer {
    public static void main(String[] args) {
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        plotErrorFunction(path, 0.16, 0, 25);

    }

    @SuppressWarnings("serial")
    public static RegressionModelBuilder getModel(String path, int iter, double stepSize, double regParam) {
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
                .set("spark.driver.allowMultipleContexts", "true");

        // Load and parse the data
        ParseCSVData data = new ParseCSVData(conf, path);

        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double v = features.toArray()[0] / 1000;
                return new LabeledPoint(point.label() / 1000, Vectors.dense(v));
            }
        });

        // building a model
        RegressionModelBuilder builder = new RegressionModelBuilder(parsedData, iter, stepSize, regParam);
        return builder;
    }

    public static void plotErrorFunction(String path, double stepsize, double regParam, int iter) {
        double[] x = new double[iter + 1];
        double[] y = new double[iter + 1];
        double[] z = new double[iter + 1];
        for (int i = 0; i <= iter; i++) {
            RegressionModelBuilder builder = getModel(path, i, stepsize, regParam);
            LinearRegressionModel model = builder.model;
            x[i] = model.intercept();
            y[i] = model.weights().toArray()[0];
            z[i] = builder.getLeastMeanSquareError();
        }

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot3DPanel plot = new Plot3DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, x, y, z);
        plot.setAxisLabels("thetha 0", "thetha 1", "error");

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("3D Road-Map Plot");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }

}
