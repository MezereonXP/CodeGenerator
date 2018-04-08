import java.util.ArrayList;

/**
 * Created by Administrator on 2018/3/20.
 */
public class MLPTest {
    public static void main(String[] args){
        int[] layer = {2,10,5,2,1};
        MLP mlp = new MLP(layer,0.005);

        int count=0;


        while(count<100000){
            ArrayList<Double> arrayList = new ArrayList<>();
            arrayList.add(Math.random()>0.5?1.0:0.0);
            arrayList.add(Math.random()>0.5?1.0:0.0);
            ArrayList<Double> tar = new ArrayList<>();
            tar.add((double) ((arrayList.get(0).intValue())&(arrayList.get(1).intValue())));
            mlp.train(arrayList,tar);
            count++;
        }
        System.out.println();



    }
}
