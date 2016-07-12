package com.learn.ml.classification.util;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.opencsv.CSVReader;

/**
 * Parsed Data creator
 * 
 * The CSV file should have the label at the last column. All the data points should be numbers
 * except for the first row(which should be name of the features).There should not be any missing
 * vales in data
 *
 */

public class CSVFileParser implements Serializable {

    private static final long serialVersionUID = 2098758145759573937L;

    private final char separator;
    private final File file;
    private List<LabeledPoint> parsedData = new ArrayList<LabeledPoint>();
    private String[] features;

    private String labeledFeature;


    public CSVFileParser(String filePath, char separator) throws IOException {
        this.separator = separator;
        this.file = new File(filePath);
        readCSVFile();
    }

    public List<LabeledPoint> getparsedData() {
        return parsedData;
    }

    public String[] getFeatures() {
        return features;
    }

    public String getLabeledFeature() {
        return labeledFeature;
    }

    private void readCSVFile() throws IOException {
        CSVReader csvReader = new CSVReader(new FileReader(file), separator);
        String[] nextLine;
        boolean firstLine = true;
        while ((nextLine = csvReader.readNext()) != null && nextLine.length > 0) {
            if (firstLine) {
                readFeatures(nextLine);
                firstLine = false;
                continue;
            }
            readDataPoint(nextLine);
        }
        csvReader.close();
    }

    private void readFeatures(String[] strings) {
        features = new String[strings.length - 1];
        // adding features
        for (int i = 0; i < features.length; i++)
            features[i] = strings[i];
        labeledFeature = strings[strings.length - 1];
    }

    private void readDataPoint(String[] dataPoint) {

        double[] v = new double[dataPoint.length - 1];

        for (int i = 0; i < dataPoint.length - 1; i++)
            v[i] = Double.parseDouble(dataPoint[i]);

        double yvalue = Double.parseDouble(dataPoint[dataPoint.length - 1]);
        parsedData.add(new LabeledPoint(yvalue, Vectors.dense(v)));
    }

}
