import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by Administrator on 2018/3/6.
 */
public class MLPLayer {
    private ArrayList<Double> inputValue;

    private ArrayList<Double> outputValue;

    private double[][] weight;
    private double[] bias;
    private double rate;
    private double[] error;
    private double[] grad;

    MLPLayer(int size, int nextSize, double learningRate){
        weight = Tool.getOneMatrix(size,nextSize);
        bias = Tool.getOneArray(nextSize);
        rate = learningRate;
        error = new double[nextSize];
        grad = new double[size];
    }

    public void calculate(){
        ArrayList<Double> out = new ArrayList<>();
        for (int i=0;i<weight[0].length;i++){
            double sum = 0;
            for (int j=0;j<inputValue.size();j++){
                sum += weight[j][i]*inputValue.get(j);
            }
            sum += bias[i];
            out.add(Tool.tanh(sum));
        }
        outputValue = out;
    }

    public void calculateError(ArrayList<Double> target) {

        double sum = 0 ;
        double[] g = new double[bias.length];
        for (int i=0;i<bias.length;i++) {
            error[i] = target.get(i)-outputValue.get(i);
            sum+=error[i];
            g[i] = error[i]*Tool.tanhD(outputValue.get(i));
            for (int j = 0; j < inputValue.size(); j++) {
                weight[j][i] += rate*error[i]*Tool.tanhD(outputValue.get(i))*inputValue.get(j);
            }
            bias[i] += rate*error[i]*Tool.tanhD(outputValue.get(i));
        }
        Arrays.fill(grad,0);
        for (int i=0;i<inputValue.size();i++){
            for (int j=0;j<bias.length;j++){
                grad[i]+=g[j]*weight[i][j];
            }
        }
        System.out.println("Loss : "+sum);
    }

    public void calculateErrorSpecial(double[] g) {
        double[] newg = new double[bias.length];
        for (int i=0;i<bias.length;i++) {

            newg[i] = g[i]*Tool.tanhD(outputValue.get(i));
            for (int j = 0; j < inputValue.size(); j++) {
                weight[j][i] += rate*g[i]*Tool.tanhD(outputValue.get(i))*inputValue.get(j);
            }
            bias[i] += rate*Tool.tanhD(outputValue.get(i));
        }
        Arrays.fill(grad,0);
        for (int i=0;i<inputValue.size();i++){
            for (int j=0;j<bias.length;j++){
                grad[i]+=newg[j]*weight[i][j];
            }
        }
    }


    public ArrayList<Double> getInputValue() {
        return inputValue;
    }

    public void setInputValue(ArrayList<Double> inputValue) {
        this.inputValue = inputValue;
    }

    public ArrayList<Double> getOutputValue() {
        return outputValue;
    }

    public void setOutputValue(ArrayList<Double> outputValue) {
        this.outputValue = outputValue;
    }

    public double[][] getWeight() {
        return weight;
    }

    public void setWeight(double[][] weight) {
        this.weight = weight;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public double getRate() {
        return rate;
    }

    public void setRate(double rate) {
        this.rate = rate;
    }

    public double[] getError() {
        return error;
    }

    public void setError(double[] error) {
        this.error = error;
    }

    public double[] getGrad() {
        return grad;
    }

    public void setGrad(double[] grad) {
        this.grad = grad;
    }
}
