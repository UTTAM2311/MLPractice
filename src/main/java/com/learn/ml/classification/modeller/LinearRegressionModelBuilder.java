package com.learn.ml.classification.modeller;

import java.io.Serializable;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.optimization.SquaredL2Updater;
import org.apache.spark.mllib.optimization.Updater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class LinearRegressionModelBuilder implements Serializable {
    private static final long serialVersionUID = 6200138757549471043L;

    public final LinearRegressionModel model;
    private JavaRDD<LabeledPoint> parsedData;

    /**
     * 
     * @param parsedData
     * @param tolerance
     * @param stepSize
     */
    public LinearRegressionModelBuilder(JavaRDD<LabeledPoint> parsedData, double tolerance, double stepSize) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        algorithm.optimizer().setStepSize(stepSize).setConvergenceTol(tolerance);
        model = algorithm.run(parsedData.rdd());
    }

    /**
     * 
     * @param parsedData
     * @param tolerance
     * @param stepSize
     * @param regParam
     */
    public LinearRegressionModelBuilder(JavaRDD<LabeledPoint> parsedData, double tolerance, double stepSize,
            double regParam) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        Updater update = new SquaredL2Updater();
        algorithm.optimizer().setStepSize(stepSize).setUpdater(update).setConvergenceTol(tolerance)
                .setRegParam(regParam);
        model = algorithm.run(parsedData.rdd());
    }

    /**
     * 
     * @param parsedData
     * @param iters
     * @param stepSize
     */
    public LinearRegressionModelBuilder(JavaRDD<LabeledPoint> parsedData, int iters, double stepSize) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        algorithm.optimizer().setStepSize(stepSize).setNumIterations(iters);
        model = algorithm.run(parsedData.rdd());
    }

    /**
     * 
     * @param parsedData
     * @param iters
     * @param stepSize
     * @param regParam
     */
    public LinearRegressionModelBuilder(JavaRDD<LabeledPoint> parsedData, int iters, double stepSize, double regParam) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        Updater update = new SquaredL2Updater();
        algorithm.optimizer().setStepSize(stepSize).setUpdater(update).setNumIterations(iters).setRegParam(regParam);
        model = algorithm.run(parsedData.rdd());
    }

    /**
     * 
     * @return
     */
    private JavaRDD<Tuple2<Double, Double>> getValuesAndPredict() {
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds =
                parsedData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    private static final long serialVersionUID = 1L;

                    public Tuple2<Double, Double> call(LabeledPoint point) {
                        double prediction = model.predict(point.features());
                        double lab = point.label();
                        return new Tuple2<Double, Double>(prediction, lab);
                    }
                });
        return valuesAndPreds;

    }

    /**
     * 
     * @return
     */
    public double getLeastMeanSquareError() {
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds = getValuesAndPredict();

        double MSE = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {
            private static final long serialVersionUID = 1L;

            public Object call(Tuple2<Double, Double> pair) {
                return Math.pow(pair._1() - pair._2(), 2.0);
            }
        }).rdd()).mean();
        return Math.sqrt(MSE);
    }

    /**
     * Mean of the Variation in actual - predicted values.
     * 
     * @return
     */
    public double getVariation() {
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds = getValuesAndPredict();

        double Mvar = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {
            private static final long serialVersionUID = 1L;

            public Object call(Tuple2<Double, Double> pair) {
                return Math.abs(pair._1() - pair._2());
            }
        }).rdd()).mean();
        return Mvar;
    }

    /**
     * 
     * @return
     */
    public String getEquation() {
        double[] coeff = model.weights().toArray();
        double intercept = model.intercept();
        StringBuilder builder = new StringBuilder();
        builder.append(intercept + " + ");
        for (int i = 0; i < coeff.length; i++) {
            builder.append(coeff[i] + "X" + (i + 1) + " + ");
        }
        builder.delete(builder.length() - 3, builder.length());
        return builder.toString();
    }
}
