package com.learn.ml.classification;

public class Tester {

    public static void main(String[] args) {
        String path = "src/main/resources/predict.csv"; // Should be some file on your system
        int order = 1;
        PolynomialRegression.drawPlots(path, order, 150, 0.5, 0);
    }
}
