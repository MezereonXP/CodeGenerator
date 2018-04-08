import java.util.ArrayList;

/**
 * Created by Administrator on 2018/3/6.
 */
public class MLP {
    private ArrayList<MLPLayer> layers;
    private double learningRate;

    MLP(int[] struct, double rate){
        layers = new ArrayList<>();
        learningRate = rate;
        for (int i=0;i<struct.length-1;i++){
            layers.add(new MLPLayer(struct[i],struct[i+1],rate));
        }
    }

    public void train(ArrayList<Double> input, ArrayList<Double> target){
        forward(input);
        backpropagation(target);
    }

    public ArrayList<Double> getResult(ArrayList<Double> input){
        forward(input);
        ArrayList<Double> result = layers.get(layers.size()-1).getOutputValue();
        return result;
    }

    private void backpropagation(ArrayList<Double> target) {

        for (int i=layers.size()-1;i>=0;i--){
            if (i==layers.size()-1) {
                layers.get(i).calculateError(target);
            }else {
                layers.get(i).calculateErrorSpecial(layers.get(i+1).getGrad());
            }
        }
    }

    private void forward(ArrayList<Double> input) {

        for (int i=0;i<layers.size();i++){
            if (i==0){
                layers.get(i).setInputValue(input);
            }else{
                layers.get(i).setInputValue(layers.get(i-1).getOutputValue());
            }
            layers.get(i).calculate();
        }
    }


    public ArrayList<MLPLayer> getLayers() {
        return layers;
    }

    public void setLayers(ArrayList<MLPLayer> layers) {
        this.layers = layers;
    }
}
