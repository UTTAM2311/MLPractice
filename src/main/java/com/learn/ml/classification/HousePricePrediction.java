package com.learn.ml.classification;

import java.awt.Color;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.math.plot.Plot2DPanel;

import com.learn.ml.classification.modeller.LinearRegressionModelBuilder;
import com.learn.ml.classification.modeller.ParseCSVData;

public class HousePricePrediction {


    public static void main(String[] args) {
        String path = "src/main/resources/houseprice.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(conf);
        // Load and parse the data
        ParseCSVData data = new ParseCSVData(sc, path);
        JavaRDD<LabeledPoint> parsedData = normalizeData(data);


        // building a model
        LinearRegressionModelBuilder builder = new LinearRegressionModelBuilder(parsedData, 10000, 0.005);
        LinearRegressionModel model = builder.model;
        List<LabeledPoint> dataPoints = parsedData.collect();

        for (LabeledPoint point : dataPoints) {
            System.out.println(" Actual Value: " + point.label() * 1000 + " Expected Value:"
                    + model.predict(point.features()) * 1000 + " DataPoint: " + point.features());
        }

        System.out.println(parsedData.count());
        System.out.println(Arrays.toString(data.getFeatures()));

        System.out.println("Training Root Mean Squared Error = " + builder.getLeastMeanSquareError() * 1000);
        System.out.println("Mean Variation in the errors = " + builder.getVariation() * 1000);
        System.out.println("Model Equation: " + builder.getEquation());

    }

    public static void plotRegularizationVariation(JavaRDD<LabeledPoint> parsedData, int iter, double stepsize) {
        double[] reg = {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] error = new double[reg.length];

        for (int i = 0; i < reg.length; i++) {
            LinearRegressionModelBuilder builder = new LinearRegressionModelBuilder(parsedData, iter, stepsize, reg[i]);
            error[i] = builder.getLeastMeanSquareError();
        }

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot2DPanel plot = new Plot2DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, reg, error);
        plot.setAxisLabels("reg", "error");

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("regParam Vs Error");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }

    @SuppressWarnings({"serial"})
    private static JavaRDD<LabeledPoint> normalizeData(ParseCSVData data) {
        JavaRDD<LabeledPoint> datam = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double[] v = new double[features.size()];
                for (int i = 0; i < v.length; i++) {
                    v[i] = features.toArray()[i] / 1000;
                }
                return new LabeledPoint(point.label() / 1000, Vectors.dense(v));
            }
        });
        return datam;
    }
}


