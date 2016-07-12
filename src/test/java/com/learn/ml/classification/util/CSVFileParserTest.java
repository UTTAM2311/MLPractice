package com.learn.ml.classification.util;

import java.io.IOException;

import org.junit.Test;

public class CSVFileParserTest {
    String filePath = "src/test/resources/sampleData.csv";
    char separator = ',';
    
    @Test
    public void test() throws IOException{
        CSVFileParser dataParser = new CSVFileParser(filePath, separator);
        System.out.println(dataParser.getLabeledFeature());
    }
}
