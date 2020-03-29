
package net.semanticmetadata.lire.imageanalysis.features;

import net.semanticmetadata.lire.utils.MetricsUtils;
import net.semanticmetadata.lire.utils.SerializationUtils;

/**
 * Generic double[] based feature implementation.
 * @author Mathias Lux, mathias@juggle.at, 27.09.13 17:00
 */
public class GenericDoubleLireFeature implements LireFeature {
    private double[] data = null;
    private String featureName = "GenericDoubleFeature";
    private String fieldName = "featGenericDouble";

    @Override
    public String getFeatureName() {
        return featureName;
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

//    @Override
//    public void extract(BufferedImage image) {
//        throw new UnsupportedOperationException("Extraction not supported.");
//    }

    @Override
    public byte[] getByteArrayRepresentation() {
        if (data == null) throw new UnsupportedOperationException("You need to set the histogram first.");
        return SerializationUtils.toByteArray(data);
    }

    @Override
    public void setByteArrayRepresentation(byte[] featureData) {
        setByteArrayRepresentation(featureData, 0, featureData.length);
    }

    @Override
    public void setByteArrayRepresentation(byte[] featureData, int offset, int length) {
        data = SerializationUtils.toDoubleArray(featureData, offset, length);
    }

    @Override
    public double[] getFeatureVector() {
        return data;
    }

    @Override
    public double getDistance(LireFeature feature) {
        // it is assumed that the histograms are of equal length.
        assert(feature.getFeatureVector().length == data.length);
        return MetricsUtils.distL2(feature.getFeatureVector(), data);
    }

//    @Override
//    public String getStringRepresentation() {
//        if (data == null) throw new UnsupportedOperationException("You need to set the histogram first.");
//        return SerializationUtils.toString(data);
//    }
//
//    @Override
//    public void setStringRepresentation(String featureVector) {
//        data = SerializationUtils.toDoubleArray(featureVector);
//    }

    public void setData(double[] data) {
        this.data = data;
    }

    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    public void getFeatureName(String featureName) {
        this.featureName = featureName;
    }
}
