package com.learn.ml.classification;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.math.plot.Plot3DPanel;



public class RoadMapModel {

    public static void main(String[] args) {
        String filepath = "src/main/resources/test.csv";
        // String filepath = "src/main/resources/3D_spatial_network.csv"; // Should be some file on
        // your system
        SparkConf conf = new SparkConf().setAppName("RoadMap Application").setMaster("local[*]");

        ParseCSVData data = new ParseCSVData(conf, filepath);
        RegressionModelBuilder builder = new RegressionModelBuilder(data.getparsedData(), 0.0001, 0.000001);


        for (LabeledPoint point : data.getparsedData().collect()) {
            System.out.println(" Actual Value: " + point.label() + " Expected Value:"
                    + builder.model.predict(point.features()) + " DataPoint: " + point.features());
        }
        System.out.println("Features: " + Arrays.toString(data.getFeatures()));
        System.out.println("LMSE: " + builder.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors = " + builder.getVariation());
        System.out.println("Coefficients: " + builder.model.weights());
        System.out.println("Intercept:" + builder.model.intercept());
        System.out.println("Equation:" + builder.getEquation());
        data.getparsedData().cache();
        int len = data.getparsedData().collect().size();
        List<String> listX = new ArrayList<>(len);
        List<String> listY = new ArrayList<>(len);
        List<String> listZ = new ArrayList<>(len);
        List<String> listZ2 = new ArrayList<>(len);
        JavaRDD<LabeledPoint> features = data.getparsedData();

        JavaRDD<String> Xvalue = features.map(new Function<LabeledPoint, String>() {

            private static final long serialVersionUID = 1L;

            public String call(LabeledPoint point) {
                double[] pt = point.features().toArray();
                return String.valueOf(pt[0]);
            }
        });
        JavaRDD<String> Yvalue = features.map(new Function<LabeledPoint, String>() {

            private static final long serialVersionUID = 1L;

            public String call(LabeledPoint point) {
                double[] pt = point.features().toArray();
                return String.valueOf(pt[1]);
            }
        });

        JavaRDD<String> Zvalue = features.map(new Function<LabeledPoint, String>() {

            private static final long serialVersionUID = 1L;

            public String call(LabeledPoint point) {
                return String.valueOf(point.label());
            }
        });
        JavaRDD<String> ZPredictvalue = features.map(new Function<LabeledPoint, String>() {

            private static final long serialVersionUID = 1L;

            public String call(LabeledPoint point) {
                double predict = builder.model.predict(point.features());
                return String.valueOf(predict);
            }
        });


        /*
         * features.foreach(new VoidFunction<LabeledPoint>() { private static final long
         * serialVersionUID = 1L;
         * 
         * @Override public void call(LabeledPoint point) throws Exception { double predict =
         * builder.model.predict(point.features()); double[] pt = point.features().toArray();
         * listX.add(String.valueOf(pt[0])); listY.add(String.valueOf(pt[1]));
         * listZ.add(String.valueOf(point.label())); listZ2.add(String.valueOf(predict)); } });
         */ features.collect();


        double[] x = new double[Xvalue.collect().size()];
        double[] y = new double[Yvalue.collect().size()];
        double[] z1 = new double[Zvalue.collect().size()];
        double[] z2 = new double[ZPredictvalue.collect().size()];

        for (int i = 0; i < x.length; i++) {
            z1[i] = Double.parseDouble(Zvalue.collect().get(i));
            z2[i] = Double.parseDouble(ZPredictvalue.collect().get(i));
            y[i] = Double.parseDouble(Zvalue.collect().get(i));
            x[i] = Double.parseDouble(Zvalue.collect().get(i));
        }



        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
        Plot3DPanel plot = new Plot3DPanel("SOUTH");

        // add grid plot to the PlotPanel
        plot.addScatterPlot("actual-plot", Color.RED, x, y, z1);
        plot.addScatterPlot("predict-plot", Color.green, x, y, z2);
        plot.setAxisLabels("latit", "long", "altit");
        plot.setBounds(7, 40, 10, 60);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }

}
