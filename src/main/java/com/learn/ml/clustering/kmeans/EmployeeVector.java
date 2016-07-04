package com.learn.ml.clustering.kmeans;

import java.util.Map;

import com.learn.ml.clustering.Vector;

/**
 * Class EmployeeVector.
 */
public class EmployeeVector extends Vector {

    private final Map<String, String> data;
    private final String empId;

    public EmployeeVector(String empId, Map<String, String> data, double[] vector) {
        super(vector);
        this.empId = empId;
        this.data = data;
    }


    public String getEmpId() {
        return empId;
    }


    public String getName() {
        return data.get("Name");
    }

    @Override
    public String toString() {
        return new org.apache.commons.lang3.builder.ToStringBuilder(this).append(super.toString())
                .append("empId", empId).toString();
    }
}
