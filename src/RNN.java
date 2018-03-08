import java.util.ArrayList;

/**
 * Created by Administrator on 2018/2/26.
 */
public class RNN {
    private double U, W, V, bias, c;
    private ArrayList<RNNCell> network;
    private double step = 1.3;

    private int inputSize, outputSize;
    public double loss;

    RNN(int inputSize, int outputSize){
        initPara();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        initNetwork(inputSize);
    }

    public void train(ArrayList data){
        forward(data);
        backPropagation(data);
    }

    public void getResult(ArrayList data){
        forward(data);
        String result = "";
        for (int i= (inputSize-outputSize);i<(inputSize);i++){
            RNNCell cell = network.get(i);
            result += cell.getOutput()+"\t";
        }
        System.out.println("Result is "+ result);
    }

    private void forward(ArrayList data) {
        for (int i = 0;i<inputSize;i++){
            RNNCell cell = network.get(i);
            if (i != 0){
                cell.setInputH(network.get(i-1).getH());
            }
            cell.inputData((Double) data.get(i), RNN.this);
            cell.calOutput(RNN.this);
        }
    }
    private void backPropagation(ArrayList data) {
        loss = 0;
        double sum = 0;
        double sumForV = 0;
        double sumForW = 0;
        double sumForB = 0;
        double sumForU = 0;

        int pos = data.size()-1;
        for (int i=(network.size()-1);i>=(inputSize-outputSize);i--){
            RNNCell cell = network.get(i);
            loss += Math.pow(cell.getOutput() - (double)data.get(pos),2);
            double temp = cell.getOutput() - (double)data.get(pos);
            sum += temp*cell.getOutput()*(1-cell.getOutput());
            sumForV += temp*cell.getOutput()*(1-cell.getOutput())*cell.getH();
            sumForW += temp*cell.getOutput()*(1-cell.getOutput())*V*(1-Math.pow(cell.getH(),2))*network.get(i-1).getH();
            sumForB += temp*cell.getOutput()*(1-cell.getOutput())*V*(1-Math.pow(cell.getH(),2));
            sumForU += temp*cell.getOutput()*(1-cell.getOutput())*V*(1-Math.pow(cell.getH(),2))*cell.getInput();
            pos--;
        }
        loss = loss/2.0;
        System.out.println("Loss : "+loss);
        this.c -= step*sum;
        this.V -= step*sumForV;
        this.W -= step*sumForW;
        this.U -= step*sumForU;
        this.bias -= step*sumForB;

    }

    private void initNetwork(int totalSize) {
        network = new ArrayList<>();
        for (int i=0;i<totalSize;i++){
            RNNCell cell = new RNNCell();
            network.add(cell);
        }
    }

    private void initPara() {
        U = 0.5;
        W = 0.5;
        V = 0.5;
        bias = 1;
        c = 1;
    }

    public double getU() {
        return U;
    }

    public void setU(double u) {
        U = u;
    }

    public double getW() {
        return W;
    }

    public void setW(double w) {
        W = w;
    }

    public double getV() {
        return V;
    }

    public void setV(double v) {
        V = v;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getC() {
        return c;
    }

    public void setC(double c) {
        this.c = c;
    }

    public ArrayList<RNNCell> getNetwork() {
        return network;
    }

    public void setNetwork(ArrayList<RNNCell> network) {
        this.network = network;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public void setOutputSize(int outputSize) {
        this.outputSize = outputSize;
    }
}
