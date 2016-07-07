package com.learn.ml.classification;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Parsed Data creator
 * 
 * The CSV file should have the label at the last column. All the data points should be numbers
 * except for the first row(which should be name of the features).There should not be any missing
 * vales in data
 * 
 * @author uttam
 *
 */

public class ParseCSVData implements Serializable {

    private static final long serialVersionUID = 2098758145759573937L;

    private JavaRDD<LabeledPoint> parsedData;
    private String[] features;

    private String labeledFeature;

    /**
     * The CSV file should have the label at the last column at the file
     * 
     * @param conf configuration for a spark application
     * @param filepath CSV filepath
     */
    public ParseCSVData(SparkConf conf, String filepath) {
        JavaSparkContext sc = new JavaSparkContext(conf);
        init(sc, filepath);
    }

    @SuppressWarnings("serial")
    private void init(JavaSparkContext sc, String filepath) {
        JavaRDD<String> data = sc.textFile(filepath);
        String[] first = data.first().split(",");
        features = new String[first.length - 1];
        // adding features
        for (int i = 0; i < features.length; i++)
            features[i] = first[i];

        // adding label
        labeledFeature = first[features.length];

        // removing the features row from the data.
        JavaRDD<String> fir = sc.parallelize(Arrays.asList(data.first()));
        JavaRDD<String> datam = data.subtract(fir);

        // Creating labeled points
        parsedData = datam.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] parts = line.split(",");
                double[] v = new double[parts.length - 1];

                for (int i = 0; i < parts.length - 1; i++)
                    v[i] = Double.parseDouble(parts[i]);

                double yvalue = Double.parseDouble(parts[parts.length - 1]);
                return new LabeledPoint(yvalue, Vectors.dense(v));
            }
        });
    }

    public JavaRDD<LabeledPoint> getparsedData() {
        return parsedData;
    }

    public String[] getFeatures() {
        return features;
    }

    public String getLabeledFeature() {
        return labeledFeature;
    }
}
