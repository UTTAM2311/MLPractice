package com.learn.ml.classification;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.parquet.it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import au.com.bytecode.opencsv.CSVReader;

public class Simple {

    @SuppressWarnings("resource")
    public static void main(String[] arg) throws IOException {
        SimpleRegression reg = new SimpleRegression(true);
        OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
        ols.setNoIntercept(false);
        CSVReader reader = new CSVReader(new FileReader(new File("src/main/resources/filteredHome.csv")));
        List<String[]> lines = reader.readAll();
        int dataPoints = (lines.size() - 1);
        System.out.println("Number of datapoints :" + dataPoints);
        int features = lines.get(0).length - 1;
        System.out.println("Number of features:" + features);

        double[][] olsX = new double[dataPoints][features];
        double[] olsY = new double[dataPoints];

        double[] x = new double[features];
        for (int i = 1; i <= dataPoints; i++) {
            String[] line = lines.get(i);
            x[0] = Double.parseDouble(line[2]);
            for (int j = 0; j < features; j++) {
                olsX[i - 1][j] = Double.parseDouble(line[j]);
            }
            olsY[i - 1] = Double.parseDouble(line[features]);

            reg.addObservation(x, Double.parseDouble(line[features]));
        }

        // muti variable regression
        ols.newSampleData(olsY, olsX);
        DoubleArrayList simlarityCoefficients = new DoubleArrayList(ols.estimateRegressionParameters());

        System.out.println("Variance of the model :" + ols.estimateErrorVariance());
        System.out.println("Coefficients: " + simlarityCoefficients.toString());
        System.out.println("Coefficients length :" + simlarityCoefficients.size());

        double sum = 0;
        for (int i = 0; i < dataPoints; i++) {
            double[] dat = olsX[i];
            double predict = predictValue(dat, simlarityCoefficients.elements());
            sum = sum + Math.pow(olsY[i] - predict, 2);

            System.out.println("datapoint:" + new DoubleArrayList(dat).toString() + "  predicted value:"
                    + predictValue(dat, simlarityCoefficients.elements()) + "  Actual value:" + olsY[i]);
        }

        System.out.println("LMSE: " + Math.sqrt(sum / 2 * dataPoints));
         // single variable regression
        System.out.println(
                "---------------------------------Regression for single variable ----------------------------------------");
        System.out.println("For a single feature  regression model:");
        System.out.println("Cost function error:" + Math.sqrt(reg.getMeanSquareError()));
        System.out.println("Intercept:" + reg.getIntercept());
        System.out.println("Slope:" + reg.getSlope());

    }

    public static double predictValue(double[] datapoint, double[] regressionParameters) {
        // regression param 0 is y intercept

        double sum = regressionParameters[0];
        for (int i = 0; i < regressionParameters.length - 1; i++) {
            sum = sum + regressionParameters[i + 1] * datapoint[i];
        }
        return sum;
    }
}
