import java.util.ArrayList;

/**
 * Created by Administrator on 2018/2/26.
 */
public class LSTMCell {
    private ArrayList<Double> inputX, inputH, inputC;
    private int lengthOfVector;
    private ArrayList<Double> outputH, outputC;
    private ArrayList<Double> forget, tempC, inputGate, tempOutput;
    private ArrayList<Double> output;

    LSTMCell(int lengthOfVector){
        this.lengthOfVector = lengthOfVector;
    }

    //Calculate the parameters in the cell of this LSTM cell, including three gates
    public void calculate(LSTM lstm){
        initArray();
        for (int i=0;i<inputX.size();i++){
            double t = lstm.getForgetU()[i]*inputX.get(i).doubleValue()+lstm.getForgetB()[i]
                    +lstm.getForgetW()[i]*inputH.get(i).doubleValue();
            double tForInputGate = lstm.getInputU()[i]*inputX.get(i)+lstm.getInputW()[i]*inputH.get(i)+lstm.getInputB()[i];
            double tForTempC = lstm.getTempcU()[i]*inputX.get(i)+lstm.getTempcW()[i]*inputH.get(i)+lstm.getTempcB()[i];
            double tForOutput = lstm.getTempcU()[i]*inputX.get(i)+lstm.getTempcW()[i]*inputH.get(i)+lstm.getTempcB()[i];

            forget.add(Tool.sigmoid(t));
            inputGate.add(Tool.sigmoid(tForInputGate));
            tempC.add(Math.tanh(tForTempC));
            tempOutput.add(Tool.sigmoid(tForOutput));
        }

        for (int i=0;i<inputX.size();i++){
            double t = forget.get(i)*inputC.get(i)+inputGate.get(i)*tempC.get(i);
            double tForOutputH = tempOutput.get(i)*Math.tanh(t);
            outputC.add(t);
            outputH.add(tForOutputH);
            output.add(Tool.sigmoid(lstm.getV()[i]*tForOutputH+lstm.getC()[i]));
        }


    }

    //init each gate with simple l
    private void initArray() {
        forget = new ArrayList();
        inputGate = new ArrayList<>();
        tempC = new ArrayList<>();
        outputC = new ArrayList<>();
        tempOutput = new ArrayList<>();
        outputH = new ArrayList<>();
        output = new ArrayList<>();
    }

    public ArrayList<Double> getInputX() {
        return inputX;
    }

    public void setInputX(ArrayList<Double> inputX) {
        this.inputX = inputX;
    }

    public ArrayList<Double> getInputH() {
        return inputH;
    }

    public void setInputH(ArrayList<Double> inputH) {
        this.inputH = inputH;
    }

    public ArrayList<Double> getInputC() {
        return inputC;
    }

    public void setInputC(ArrayList<Double> inputC) {
        this.inputC = inputC;
    }

    public int getLengthOfVector() {
        return lengthOfVector;
    }

    public void setLengthOfVector(int lengthOfVector) {
        this.lengthOfVector = lengthOfVector;
    }

    public ArrayList<Double> getOutputH() {
        return outputH;
    }

    public void setOutputH(ArrayList<Double> outputH) {
        this.outputH = outputH;
    }

    public ArrayList<Double> getOutputC() {
        return outputC;
    }

    public void setOutputC(ArrayList<Double> outputC) {
        this.outputC = outputC;
    }

    public ArrayList<Double> getForget() {
        return forget;
    }

    public void setForget(ArrayList<Double> forget) {
        this.forget = forget;
    }

    public ArrayList<Double> getTempC() {
        return tempC;
    }

    public void setTempC(ArrayList<Double> tempC) {
        this.tempC = tempC;
    }

    public ArrayList<Double> getInputGate() {
        return inputGate;
    }

    public void setInputGate(ArrayList<Double> inputGate) {
        this.inputGate = inputGate;
    }

    public ArrayList<Double> getTempOutput() {
        return tempOutput;
    }

    public void setTempOutput(ArrayList<Double> tempOutput) {
        this.tempOutput = tempOutput;
    }

    public ArrayList<Double> getOutput() {
        return output;
    }

    public void setOutput(ArrayList<Double> output) {
        this.output = output;
    }
}
