/**
 * Created by Administrator on 2018/2/26.
 */
public class RNNCell {
    private double inputH;
    private double h;
    private double input;

    private double output = -1;//If this cell is not the cell of outputLayer, set it value as -1
    RNNCell(){
        h = 0;
        inputH = 0;
        input = 0;
    }

    public void inputData(double x, RNN rnn){
        input = x;
        h = Math.tanh(rnn.getU()*input+rnn.getW()*inputH+rnn.getBias());
    }

    //using sigmoid
    public void calOutput(RNN rnn){
        this.output = 1.0/(1+Math.exp(-1*(rnn.getV()*this.h+rnn.getC())));
    }

    public double getInputH() {
        return inputH;
    }

    public void setInputH(double inputH) {
        this.inputH = inputH;
    }

    public double getH() {
        return h;
    }

    public void setH(double h) {
        this.h = h;
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}
