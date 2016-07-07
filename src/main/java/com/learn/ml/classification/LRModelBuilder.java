package com.learn.ml.classification;

import java.io.Serializable;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class LRModelBuilder implements Serializable {
    private static final long serialVersionUID = 6200138757549471043L;

    public final LinearRegressionModel model;
    private JavaRDD<LabeledPoint> parsedData;

    public LRModelBuilder(JavaRDD<LabeledPoint> parsedData, double tolerance, double stepSize) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        algorithm.optimizer().setStepSize(stepSize).setConvergenceTol(tolerance);
        model = algorithm.run(parsedData.rdd());
    }

    public LRModelBuilder(JavaRDD<LabeledPoint> parsedData, int iters, double stepSize) {
        this.parsedData = parsedData;
        LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
        algorithm.setIntercept(true);
        algorithm.optimizer().setStepSize(stepSize).setNumIterations(iters);
        model = algorithm.run(parsedData.rdd());
    }

    public double getLeastMeanSquareError() {
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds =
                parsedData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    private static final long serialVersionUID = 1L;

                    public Tuple2<Double, Double> call(LabeledPoint point) {
                        double prediction = model.predict(point.features());
                        double lab = point.label();
                        return new Tuple2<Double, Double>(prediction, lab);
                    }
                });

        double MSE = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {

            private static final long serialVersionUID = -7595654429741861559L;

            public Object call(Tuple2<Double, Double> pair) {
                return Math.pow(pair._1() - pair._2(), 2.0);
            }
        }).rdd()).mean();

        return Math.sqrt(MSE);
    }
}
