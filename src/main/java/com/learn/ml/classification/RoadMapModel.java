package com.learn.ml.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.regression.LabeledPoint;



public class RoadMapModel {

    public static void main(String[] args) {
        String filepath = "src/main/resources/3D_spatial_network.csv"; // Should be some file on
                                                                       // your system
        SparkConf conf = new SparkConf().setAppName("RoadMap Application").setMaster("local[*]");

        ParseCSVData data = new ParseCSVData(conf, filepath);
        RegressionModelBuilder builder = new RegressionModelBuilder(data.getparsedData(), 0.0001, 0.000001);


        System.out.println("Features: " + Arrays.toString(data.getFeatures()));
        System.out.println("LMSE: " + builder.getLeastMeanSquareError());
        System.out.println("Mean Variation in the errors = " + builder.getVariation());
        System.out.println("Coefficients: " + builder.model.weights());
        System.out.println("Intercept:" + builder.model.intercept());
        System.out.println("Equation:" + builder.getEquation());

        List<String> listX = new ArrayList<>();
        List<String> listY = new ArrayList<>();
        List<String> listZ = new ArrayList<>();
        List<String> listZ2 = new ArrayList<>();
        JavaRDD<LabeledPoint> features = data.getparsedData();
        features.foreach(new VoidFunction<LabeledPoint>() {
            private static final long serialVersionUID = 1L;

            @Override
            public void call(LabeledPoint point) throws Exception {
                double predict = builder.model.predict(point.features());
                double[] pt = point.features().toArray();
                listX.add(String.valueOf(pt[0]));
                listY.add(String.valueOf(pt[1]));
                listZ.add(String.valueOf(point.label()));
                listZ2.add(String.valueOf(predict));
            }
        });

        double[] x = new double[listX.size()];
        double[] y = new double[listY.size()];
        double[] z1 = new double[listX.size()];
        double[] z2 = new double[listX.size()];

        for (int i = 0; i < x.length; i++) {
            z1[i] = Double.parseDouble(listZ.get(i));
            z2[i] = Double.parseDouble(listZ2.get(i));
            y[i] = Double.parseDouble(listY.get(i));
            x[i] = Double.parseDouble(listX.get(i));
        }

        /*
         * // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH Plot3DPanel
         * plot = new Plot3DPanel();
         * 
         * // add grid plot to the PlotPanel plot.addScatterPlot("plot", x, y, z1);
         * plot.setAxisLabels("latit", "long", "altit");
         * 
         * // put the PlotPanel in a JFrame like a JPanel JFrame frame = new JFrame("a plot panel");
         * frame.setSize(600, 600); frame.setContentPane(plot); frame.setVisible(true);
         */
    }

}
