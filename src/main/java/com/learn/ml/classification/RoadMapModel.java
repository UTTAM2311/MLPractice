package com.learn.ml.classification;

import java.awt.Color;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.math.plot.Plot3DPanel;



public class RoadMapModel {

    public static void main(String[] args) {
        String filepath = "src/main/resources/3D_spatial_network.csv"; // Should be some file on
        // your system
        SparkConf conf = new SparkConf().setAppName("RoadMap Application").setMaster("local[*]");

        ParseCSVData data = new ParseCSVData(conf, filepath);
        RegressionModelBuilder builder = new RegressionModelBuilder(data.getparsedData(), 0.0001, 0.0005);
        List<LabeledPoint> dataPoints = data.getparsedData().collect();


        for (LabeledPoint point : data.getparsedData().collect()) {
            System.out.println(" Actual Value: " + point.label() + " Expected Value:"
                    + builder.model.predict(point.features()) + " DataPoint: " + point.features());
        }


        System.out.println(dataPoints.size());
        System.out.println("Features: " + Arrays.toString(data.getFeatures()));
        System.out.println("LMSE: " + builder.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors = " + builder.getVariation());
        System.out.println("Coefficients: " + builder.model.weights());
        System.out.println("Intercept:" + builder.model.intercept());
        System.out.println("Equation:" + builder.getEquation());
        data.getparsedData().cache();


        int len = data.getparsedData().collect().size();
        JavaRDD<LabeledPoint> features = data.getparsedData();

        double[] x = new double[len];
        double[] y = new double[len];
        double[] z1 = new double[len];
        double[] z2 = new double[len];


        for (int i = 0; i < dataPoints.size(); i++) {
            LabeledPoint point = dataPoints.get(i);
            double predict = builder.model.predict(point.features());
            double[] pt = point.features().toArray();
            x[i] = pt[0];
            y[i] = pt[1];
            z1[i] = point.label();
            z2[i] = predict;
        }

        features.collect();

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot3DPanel plot = new Plot3DPanel();

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, x, y, z1);
        plot.addScatterPlot("predict-plot", Color.green, x, y, z2);
        plot.setAxisLabels("latit", "long", "altit");
        plot.setBounds(8, 45, 10, 60);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("3D Road-Map Plot");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }

}
