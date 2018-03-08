import java.util.ArrayList;

/**
 * Created by Administrator on 2018/2/27.
 */
public class Tool {
    public static double sigmoid(double x){
        return 1.0/(1+Math.exp(-1*x));
    }
    public static ArrayList getEmptyArray(int length){
        ArrayList<Double> list = new ArrayList<>();
        for (int i=0;i<length;i++){
            list.add(0.0);
        }
        return list;
    }

    public static double getDifference(ArrayList<Double> a, ArrayList<Double> b){
        double sum = 0;
        for (int i=0;i<a.size();i++){
            sum += a.get(i)-b.get(i);
        }
        return sum;
    }

    public static double[] getOneArray(int length){
        double[] a = new double[length];
        for (int i=0;i<length;i++){
            a[i] = 1.0;
        }
        return a;
    }

    public static double[] getSpecialArray(int length, double value){
        double[] a = new double[length];
        for (int i=0;i<length;i++){
            a[i] = value;
        }
        return a;
    }

    public static double[][] getZeroMatrix(int x, int y){
        double[][] matrix = new double[x][y];
        for (int i=0;i<x;i++){
            for (int j=0;j<y;j++){
                matrix[i][j] = 0.0;
            }
        }
        return matrix;
    }
}
