package com.learn.ml.classification;

import java.awt.Color;
import java.util.List;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.math.plot.Plot2DPanel;

import com.learn.ml.classification.modeller.ParseCSVData;
import com.learn.ml.classification.modeller.PolynomialRegressionModelBuilder;

public class PolyRegHigherStudy {

    public static void main(String[] args) {
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        plotOrder(sc, path);
        plotErrorVsOrder(sc, path);
    }

    private static void plotErrorVsOrder(JavaSparkContext sc, String path) {
        double[] error = new double[5];
        double[] ord = new double[5];
        ord[0] = 1;
        ord[1] = 2;
        ord[2] = 3;
        ord[3] = 4;
        ord[4] = 5;
        error[0] = getError(sc, path, 1, 10000, 0.155, 0);
        error[1] = getError(sc, path, 2, 10000, 0.155, 0);
        error[2] = getError(sc, path, 3, 10000, 0.01155, 0);
        error[3] = getError(sc, path, 4, 10000, 0.000655, 0);
        error[4] = getError(sc, path, 5, 10000, 0.0000006, 0);

        JFrame frame = new JFrame("Plots for different ordered polynomial functions");
        Plot2DPanel plot = new Plot2DPanel();
        plot2D(plot, frame, ord, error, Color.BLUE, "Error Vs Order plot");

    }

    @SuppressWarnings("serial")
    private static double getError(JavaSparkContext sc, String path, int order, int iter, double stepsize,
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
        PolynomialRegressionModelBuilder builder =
                new PolynomialRegressionModelBuilder(parsedData, order, iter, stepsize, regParam);


        return builder.getLeastMeanSquareError();
    }

    @SuppressWarnings("serial")
    public static void plotOrder(JavaSparkContext sc, String path) {

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
        PolynomialRegressionModelBuilder builder = new PolynomialRegressionModelBuilder(parsedData, 1, 10000, 0.155, 0);
        PolynomialRegressionModelBuilder builder2 =
                new PolynomialRegressionModelBuilder(parsedData, 2, 10000, 0.155, 0);
        PolynomialRegressionModelBuilder builder3 =
                new PolynomialRegressionModelBuilder(parsedData, 3, 10000, 0.01155, 0);
        PolynomialRegressionModelBuilder builder4 =
                new PolynomialRegressionModelBuilder(parsedData, 4, 10000, 0.000655, 0);
        PolynomialRegressionModelBuilder builder5 =
                new PolynomialRegressionModelBuilder(parsedData, 5, 10000, 0.0000006, 0);



        List<LabeledPoint> points = parsedData.collect();
        int len = points.size();
        double[] x = new double[len];
        double[] y1 = new double[len];
        double[] y2 = new double[len];
        double[] y3 = new double[len];
        double[] y4 = new double[len];
        double[] y5 = new double[len];
        double[] yA = new double[len];
        for (int i = 0; i < len; i++) {
            x[i] = points.get(i).features().toArray()[0];
            y1[i] = builder.model.predict(points.get(i).features());
            y2[i] = builder2.model.predict(builder2.getParsedData().collect().get(i).features());
            y3[i] = builder3.model.predict(builder3.getParsedData().collect().get(i).features());
            y4[i] = builder4.model.predict(builder4.getParsedData().collect().get(i).features());
            y5[i] = builder5.model.predict(builder5.getParsedData().collect().get(i).features());
            yA[i] = points.get(i).label();
        }

        JFrame frame = new JFrame("House plot - from polynomial regression");
        Plot2DPanel plot = new Plot2DPanel();

        plot2D(plot, frame, x, y1, Color.BLUE, "Linear Model plot");
        plot2D(plot, frame, x, y2, Color.green, "2nd degree plot");
        plot2D(plot, frame, x, y3, Color.yellow, "3rd degree plot");
        plot2D(plot, frame, x, y4, Color.CYAN, "4th degree plot");
        plot2D(plot, frame, x, y5, Color.ORANGE, "5th degree plot");
        plot2D(plot, frame, x, yA, Color.red, "Actual- points");


    }

    public static void plot2D(Plot2DPanel plot, JFrame frame, double[] x, double[] y, Color color,
            String curveDetails) {
        if (x.length != y.length)
            throw new IllegalArgumentException("Size of the x y must be equal");

        // add grid plot to the PlotPanel
        plot.addScatterPlot(curveDetails, color, x, y);
        plot.setAxisLabels("x", "y");

        // put the PlotPanel in a JFrame like a JPanel
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }
}
