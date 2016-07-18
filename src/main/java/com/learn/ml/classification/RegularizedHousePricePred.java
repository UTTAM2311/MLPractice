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
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");

        // Load and parse the data
        ParseCSVData data = new ParseCSVData(conf, path);
        JavaRDD<LabeledPoint> parsedData = data.getparsedData().map(new Function<LabeledPoint, LabeledPoint>() {
            public LabeledPoint call(LabeledPoint point) {
                Vector features = point.features();
                double[] v = new double[4];
                v[0] = features.toArray()[0] / 1000;
                v[1] = Math.pow(v[0], 2);
                v[2] = Math.pow(v[0], 3);
                v[3] = Math.pow(v[0], 4);
                /*
                 * v[4] = Math.pow(v[0], 5) / 1000; v[5] = Math.pow(v[0], 6) / 1000; v[6] =
                 * Math.pow(v[0], 7) / 1000; v[7] = Math.pow(v[0], 8) / 1000; v[8] = Math.pow(v[0],
                 * 9) / 1000; v[9] = Math.pow(v[0], 10) / 1000;
                 */

                return new LabeledPoint(point.label() / 1000, Vectors.dense(v));
            }
        });


        // building a model
        RegressionModelBuilder builder = new RegressionModelBuilder(parsedData, 0.00001, 0.000055,0.00001 );
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
