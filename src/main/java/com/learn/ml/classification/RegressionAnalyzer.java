package com.learn.ml.classification;

import java.awt.Color;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.math.plot.Plot3DPanel;

import com.learn.ml.classification.modeller.LinearRegressionModelBuilder;
import com.learn.ml.classification.modeller.ParseCSVData;

public class RegressionAnalyzer {
    public static void main(String[] args) {
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
                .set("spark.driver.allowMultipleContexts", "true");

        JavaSparkContext sc = new JavaSparkContext(conf);
        double stepSize = 0.02;
        int iter = 60;
        double regparam = 0;
        plotErrorFunction(path, sc, stepSize, regparam, iter);

    }

    @SuppressWarnings("serial")
    public static LinearRegressionModelBuilder getModel(String path, JavaSparkContext sc, int iter, double stepSize,
            double regParam) {
        // Load and parse the data
        ParseCSVData data = new ParseCSVData(sc, path);

        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double v = features.toArray()[0] / 1000;
                return new LabeledPoint(point.label() / 1000, Vectors.dense(v));
            }
        });

        // building a model
        LinearRegressionModelBuilder builder = new LinearRegressionModelBuilder(parsedData, iter, stepSize, regParam);
        return builder;
    }

    public static void plotErrorFunction(String path, JavaSparkContext sc, double stepsize, double regParam, int iter) {
        double[] thetha0 = new double[iter + 1];
        double[] thetha1 = new double[iter + 1];
        double[] error = new double[iter + 1];

        for (int i = 0; i <= iter; i++) {
            LinearRegressionModelBuilder builder = getModel(path, sc, i, stepsize, regParam);
            LinearRegressionModel model = builder.model;
            thetha0[i] = model.intercept();
            thetha1[i] = model.weights().toArray()[0];
            error[i] = builder.getLeastMeanSquareError();
        }

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot3DPanel plot = new Plot3DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, thetha0, thetha1, error);
        plot.setAxisLabels("thetha 0", "thetha 1", "error");

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("paraboloid Plot");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }

}
