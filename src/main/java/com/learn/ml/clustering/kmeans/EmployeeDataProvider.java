package com.learn.ml.clustering.kmeans;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.StringUtils;

/**
 * Class DataHolder.
 * 
 */
public class EmployeeDataProvider implements VectorDataProvider<EmployeeVector> {

    private final String fileName;

    public EmployeeDataProvider(String fileName) {
        this.fileName = fileName;
    }

    private Set<String> getCapabilityHeaders(Set<String> headers) {
        return headers.stream().filter(h -> !(h.toLowerCase().equals("name") || h.toLowerCase().equals("empid")))
                .collect(Collectors.toSet());
    }

    private double[] getVector(Set<String> headers, CSVRecord record) {
        return headers.stream().mapToDouble(s -> {
            return getScore(record.get(s));
        }).toArray();
    }

    private double getScore(String s) {
        if (StringUtils.isBlank(s) || "Want to".equalsIgnoreCase(s)) {
            return 0;
        } else if ("Beginner".equalsIgnoreCase(s) || "Intermediate".equalsIgnoreCase(s) || "Expert".equalsIgnoreCase(s)
                || "Learning".equalsIgnoreCase(s)) {
            return 1;
        }
        return 0;
    }

    private boolean isKnownTechnology(String s) {
        if (StringUtils.isBlank(s)) {
            return false;
        }
        return "Beginner".equalsIgnoreCase(s) || "Intermediate".equalsIgnoreCase(s) || "Expert".equalsIgnoreCase(s)
                || "Learning".equalsIgnoreCase(s);
    }

    @Override
    public List<EmployeeVector> getVectorData() throws IOException {
        Reader reader =
                new InputStreamReader(Thread.currentThread().getContextClassLoader().getResourceAsStream(fileName));
        List<EmployeeVector> vectors = new ArrayList<>();
        CSVParser csvParser = new CSVParser(reader, CSVFormat.EXCEL.withHeader());
        Set<String> headers = getCapabilityHeaders(csvParser.getHeaderMap().keySet());
        try {
            for (final CSVRecord record : csvParser) {
                String empId = record.get(0);
                Map<String, String> data = record.toMap();
                double[] vector = getVector(headers, record);
                EmployeeVector employeeVector = new EmployeeVector(empId, data, vector);
                vectors.add(employeeVector);
            }
        } finally {
            csvParser.close();
            reader.close();
        }
        return vectors;
    }


    public static void main(String[] args) throws Exception {
        EmployeeDataProvider employeeDataProvider = new EmployeeDataProvider("data.csv");
        List<EmployeeVector> vectorData = employeeDataProvider.getVectorData();
        System.out.println(vectorData);
    }
}
