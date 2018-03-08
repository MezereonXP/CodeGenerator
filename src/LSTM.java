import java.util.ArrayList;

/**
 * Created by Administrator on 2018/2/26.
 */
public class LSTM {
    private double[] forgetW , forgetU , forgetB ;
    private double[] inputW , inputU , inputB ;
    private double[] tempcW , tempcU , tempcB ;
    private double[] outputW , outputU , outputB ;
    private double[] V , C ;
    private double step = 10.0, stepInScore = 0.1;

    private double[] W = Tool.getSpecialArray(768,1.0/768);

    private double[][] score;

    private ArrayList<LSTMCell> network;
    private int inputSize, outputSize, lengthForEachInput;
    private double loss;
    private double lossForAll = 0;

    LSTM(int inputSize, int outputSize, int lengthForEachInput){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.lengthForEachInput = lengthForEachInput;
        network = new ArrayList<>();
        for (int i=0;i<inputSize;i++){
            LSTMCell lstmCell = new LSTMCell(lengthForEachInput);
            network.add(lstmCell);
        }
        V = Tool.getOneArray(lengthForEachInput);
        C = Tool.getOneArray(lengthForEachInput);
        forgetW = Tool.getOneArray(lengthForEachInput);
        forgetU = Tool.getOneArray(lengthForEachInput);
        forgetB = Tool.getOneArray(lengthForEachInput);
        inputW = Tool.getOneArray(lengthForEachInput);
        inputU = Tool.getOneArray(lengthForEachInput);
        inputB = Tool.getOneArray(lengthForEachInput);
        tempcW = Tool.getOneArray(lengthForEachInput);
        tempcU = Tool.getOneArray(lengthForEachInput);
        tempcB = Tool.getOneArray(lengthForEachInput);
        outputW = Tool.getOneArray(lengthForEachInput);
        outputU = Tool.getOneArray(lengthForEachInput);
        outputB = Tool.getOneArray(lengthForEachInput);
    }

    public void train(ArrayList<ArrayList> data){
        forward(data);
        backPropagation(data);
    }

    public void trainForNLP(ArrayList<ArrayList> data, double[][] realScore){
        forward(data);
        calculateError(data, realScore);
        backPropagation(data);
    }

    public void getResultForNLP(ArrayList<ArrayList> data){
        forward(data);
        ArrayList<ArrayList> out = new ArrayList<>();
        for (int i=inputSize-outputSize;i<inputSize;i++){
            ArrayList output = new ArrayList();
            for (int j=0;j<lengthForEachInput;j++){
                output.add(network.get(i).getOutput().get(j));
            }
            out.add(output);
        }

        score = new double[inputSize][inputSize];
        for (int i=0;i<inputSize;i++){
            for (int j=0;j<inputSize;j++){
                if (i == j){
                    score[i][j] = 0;
                }else{
                    score[i][j] = getScore(out, i, j);
                }
                System.out.print(score[i][j]+"\t");
            }
            System.out.println();
        }
    }

    private void calculateError(ArrayList<ArrayList> data, double[][] realScore) {

        ArrayList<ArrayList> out = new ArrayList<>();
        for (int i=inputSize-outputSize;i<inputSize;i++){
            ArrayList output = new ArrayList();
            for (int j=0;j<lengthForEachInput;j++){
                output.add(network.get(i).getOutput().get(j));
            }
            out.add(output);
        }

        score = new double[inputSize][inputSize];
        lossForAll = 0;
        for (int i=0;i<inputSize;i++){
            for (int j=0;j<inputSize;j++){
                if (i == j){
                    score[i][j] = 0;
                }else{
                    score[i][j] = getScore(out, i, j);
                    double loss = score[i][j]-realScore[i][j];
                    lossForAll += Math.abs(loss);
                    updateW(out, loss, i, j);
                    updateOut(out, loss, i, j);
                }
            }
        }
        for (ArrayList list:out){
            data.add(list);
        }
    }

    private void updateOut(ArrayList<ArrayList> out, double loss, int i, int j) {
        for (int k=0;k<768;k++){
            if (k<384)
                out.get(i).set(k, (double)out.get(i).get(k)+stepInScore*loss*W[k]);
            else
                out.get(j).set(k-384, (double)out.get(j).get(k-384)+stepInScore*loss*W[k]);
        }
    }

    private void updateW(ArrayList<ArrayList> out, double loss, int i, int j) {
        for (int k=0;k<768;k++){
            if (k<384)
                W[k] -= stepInScore*loss*(double)out.get(i).get(k);
            else
                W[k] -= stepInScore*loss*(double)out.get(j).get(k-384);
        }
    }

    private double getScore(ArrayList<ArrayList> out, int i, int j) {
        double result = 0;
        for (int k=0;k<768;k++){
            if (k<384)
                result += W[k]*(double)out.get(i).get(k);
            else
                result += W[k]*(double)out.get(j).get(k-384);
        }
        return 1.0/(1+Math.exp(-1*result));
    }

    public void getResult(ArrayList<ArrayList> data){
        forward(data);
        for (int i=inputSize-outputSize;i<inputSize;i++){
            for (int j=0;j<lengthForEachInput;j++){
                System.out.print(network.get(i).getOutput().get(j)+",");
            }
            System.out.println();
        }
    }

    private void backPropagation(ArrayList<ArrayList> data) {
        int pos = data.size()-1;
        loss = 0;
        for (int i=inputSize-1;i>=inputSize-outputSize&&i!=0;i--){
            double temp = 0;
            double sumForV = 0, sumForC = 0;
            double sumForFW = 0, sumForFU = 0, sumForFB = 0;
            double sumForIW = 0, sumForIU = 0, sumForIB = 0;
            double sumForTCW = 0, sumForTCU = 0, sumForTCB = 0;
            double sumForOW = 0, sumForOU = 0, sumForOB = 0;

            for (int j=0;j<lengthForEachInput;j++){
                double yt = network.get(i).getOutput().get(j);
                double xt = network.get(i).getInputX().get(j);
                double ht = network.get(i).getOutputH().get(j);
                double ot = network.get(i).getTempOutput().get(j);
                double ct = network.get(i).getOutputC().get(j);
                double tempct = network.get(i).getTempC().get(j);
                double ft = network.get(i).getForget().get(j);
                double it = network.get(i).getInputGate().get(j);

                double ct2 = network.get(i-1).getOutputC().get(j);
                double ht2 = network.get(i-1).getOutputH().get(j);

                double d = (yt - (double)data.get(pos).get(j))*yt*(1-yt)*V[j]*ot*(1-Math.tanh(ct)*Math.tanh(ct));

                sumForV += (yt - (double)data.get(pos).get(j))*yt*(1-yt)*ht;
                sumForC += (yt - (double)data.get(pos).get(j))*yt*(1-yt);
                temp += (yt - (double)data.get(pos).get(j));

                if (i == (inputSize-1)){
                    sumForFW += d*ct2*ft*(1-ft)*ht2;
                    sumForFU += d*ct2*ft*(1-ft)*xt;
                    sumForFB += d*ct2*ft*(1-ft);

                    sumForIW += d*tempct*it*(1-it)*ht2;
                    sumForIU += d*tempct*it*(1-it)*xt;
                    sumForIB += d*tempct*it*(1-it);

                    sumForTCW += d*it*(1-Math.tanh(tempct)*Math.tanh(tempct))*ht2;
                    sumForTCU += d*it*(1-Math.tanh(tempct)*Math.tanh(tempct))*xt;
                    sumForTCB += d*it*(1-Math.tanh(tempct)*Math.tanh(tempct));

                }else{
                    double syt = network.get(i+1).getOutputH().get(j);
                    double sot = network.get(i+1).getTempOutput().get(j);
                    double sct = network.get(i+1).getOutputC().get(j);
                    double sft = network.get(i+1).getForget().get(j);

                    double d2 = (syt - (double)data.get(pos+1).get(j))*syt*(1-syt)*V[j]
                                    *sot*(1-Math.tanh(sct)*Math.tanh(sct))*sft;

                    sumForFW += (d+d2)*ct2*ft*(1-ft)*ht2;
                    sumForFU += (d+d2)*ct2*ft*(1-ft)*xt;
                    sumForFB += (d+d2)*ct2*ft*(1-ft);

                    sumForIW += (d+d2)*tempct*it*(1-it)*ht2;
                    sumForIU += (d+d2)*tempct*it*(1-it)*xt;
                    sumForIB += (d+d2)*tempct*it*(1-it);

                    sumForTCW += (d+d2)*it*(1-Math.tanh(tempct)*Math.tanh(tempct))*ht2;
                    sumForTCU += (d+d2)*it*(1-Math.tanh(tempct)*Math.tanh(tempct))*xt;
                    sumForTCB += (d+d2)*it*(1-Math.tanh(tempct)*Math.tanh(tempct));
                }
                sumForOW += (yt - (double)data.get(pos).get(j))*yt*(1-yt)*V[j]*Math.tanh(ct)*ot*(1-ot)*ht2;
                sumForOU += (yt - (double)data.get(pos).get(j))*yt*(1-yt)*V[j]*Math.tanh(ct)*ot*(1-ot)*xt;
                sumForOB += (yt - (double)data.get(pos).get(j))*yt*(1-yt)*V[j]*Math.tanh(ct)*ot*(1-ot);

                this.forgetW[j] -= step*sumForFW;
                this.forgetU[j] -= step*sumForFU;
                this.forgetB[j] -= step*sumForFB;
                this.inputW[j] -= step*sumForIW;
                this.inputU[j] -= step*sumForIU;
                this.inputB[j] -= step*sumForIB;
                this.tempcW[j] -= step*sumForTCW;
                this.tempcU[j] -= step*sumForTCU;
                this.tempcB[j] -= step*sumForTCB;
                this.outputW[j] -= step*sumForOW;
                this.outputU[j] -= step*sumForOU;
                this.outputB[j] -= step*sumForOB;
                this.V[j] -= step*sumForV;
                this.C[j] -= step*sumForC;
            }
            loss+=Math.pow(temp,2);
            pos--;
        }
        loss = loss/2.0;
        //System.out.println("LSTM Loss is "+loss);
    }

    private void forward(ArrayList<ArrayList> data) {
        for (int i=0;i<inputSize;i++){
            LSTMCell cell = network.get(i);
            if (i == 0){
                cell.setInputC(Tool.getEmptyArray(lengthForEachInput));
                cell.setInputH(Tool.getEmptyArray(lengthForEachInput));
            }else{
                cell.setInputC(network.get(i-1).getOutputC());
                cell.setInputH(network.get(i-1).getOutputH());
            }
            cell.setInputX(data.get(i));
            cell.calculate(LSTM.this);
        }
    }



    public double[] getV() {
        return V;
    }

    public void setV(double[] v) {
        V = v;
    }

    public double[] getC() {
        return C;
    }

    public void setC(double[] c) {
        C = c;
    }

    public double getLoss() {
        return loss;
    }

    public void setLoss(double loss) {
        this.loss = loss;
    }

    public double[] getForgetW() {
        return forgetW;
    }

    public void setForgetW(double[] forgetW) {
        this.forgetW = forgetW;
    }

    public double[] getForgetU() {
        return forgetU;
    }

    public void setForgetU(double[] forgetU) {
        this.forgetU = forgetU;
    }

    public double[] getForgetB() {
        return forgetB;
    }

    public void setForgetB(double[] forgetB) {
        this.forgetB = forgetB;
    }

    public double[] getInputW() {
        return inputW;
    }

    public void setInputW(double[] inputW) {
        this.inputW = inputW;
    }

    public double[] getInputU() {
        return inputU;
    }

    public void setInputU(double[] inputU) {
        this.inputU = inputU;
    }

    public double[] getInputB() {
        return inputB;
    }

    public void setInputB(double[] inputB) {
        this.inputB = inputB;
    }

    public double[] getTempcW() {
        return tempcW;
    }

    public void setTempcW(double[] tempcW) {
        this.tempcW = tempcW;
    }

    public double[] getTempcU() {
        return tempcU;
    }

    public void setTempcU(double[] tempcU) {
        this.tempcU = tempcU;
    }

    public double[] getTempcB() {
        return tempcB;
    }

    public void setTempcB(double[] tempcB) {
        this.tempcB = tempcB;
    }

    public double[] getOutputW() {
        return outputW;
    }

    public void setOutputW(double[] outputW) {
        this.outputW = outputW;
    }

    public double[] getOutputU() {
        return outputU;
    }

    public void setOutputU(double[] outputU) {
        this.outputU = outputU;
    }

    public double[] getOutputB() {
        return outputB;
    }

    public void setOutputB(double[] outputB) {
        this.outputB = outputB;
    }

    public double getLossForAll() {
        return lossForAll;
    }

    public void setLossForAll(double lossForAll) {
        this.lossForAll = lossForAll;
    }
}
