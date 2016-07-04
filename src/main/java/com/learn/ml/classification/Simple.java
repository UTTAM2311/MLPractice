package com.learn.ml.classification;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.parquet.it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import au.com.bytecode.opencsv.CSVReader;

public class Simple {

    @SuppressWarnings("resource")
    public static void main(String[] arg) throws IOException {
        SimpleRegression reg = new SimpleRegression(true);
        OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();

        CSVReader reader = new CSVReader(new FileReader(new File("src/main/resources/filteredHome.csv")));
        List<String[]> lines = reader.readAll();
        System.out.println("Number of datapoints :" + lines.size());
        int features = lines.get(0).length - 1;
        System.out.println("Number of features:" + features);

        double[][] olsX = new double[lines.size()][features];
        double[] olsY = new double[lines.size()];

        double[] x = new double[features];
        for (int i = 1; i < lines.size(); i++) {
            String[] line = lines.get(i);
            x[0] = Double.parseDouble(line[2]);
            for (int j = 0; j < features; j++) {
                olsX[i][j] = Double.parseDouble(line[j]);
            }
            olsY[i] = Double.parseDouble(line[features]);

            reg.addObservation(x, Double.parseDouble(line[features]));
        }

        //
        ols.newSampleData(olsY, olsX);
        DoubleArrayList simlarityCoefficients = new DoubleArrayList(ols.estimateRegressionParameters());

        System.out.println("Varaiance of the model :" + ols.estimateRegressandVariance());
        System.out.println("Coefficients: " + simlarityCoefficients.toString());
        System.out.println("Coefficients length :" + simlarityCoefficients.size());
        System.out.println(ols.estimateResiduals().length);
        // simple regression
        System.out.println("For a single features  regression model:");
        System.out.println("Cost function error:" + Math.sqrt(reg.getMeanSquareError()));
        System.out.println("Intercept:" + reg.getIntercept());
        System.out.println("Slope:" + reg.getSlope());

    }
}
