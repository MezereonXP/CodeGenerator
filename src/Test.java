import com.csvreader.CsvReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by Administrator on 2018/3/6.
 */
public class Test {
    public static void main(String[] args) throws IOException {



        LSTM lstm = null;
        int count = 0;
        while(count<5) {
            for (int i=1;i<=1;i++) {
                ArrayList<ArrayList> data = getData(i);

                if (lstm==null)
                    lstm = new LSTM(data.size(),data.size(),data.get(0).size());
                else
                    lstm.changeSize(data.size(),data.size());

                double[][] score = Tool.getZeroMatrix(data.size(),data.size());
                for (int j=0;j<score.length;j++){
                    Arrays.fill(score[j],0.001);
                }
                updateScore(score, i);
                lstm.trainForNLP(data, score);
            }

            System.out.print(lstm.getLossForAll()+", ");
            count++;
        }
        System.out.println();
        ArrayList<ArrayList> data = getData(1);
        lstm.changeSize(data.size(),data.size());
        lstm.getResultForNLP(data);
    }

    private static void updateScore(double[][] score,int i) throws IOException {
        File file= new File("F:\\Project\\Software Engineer\\matrix\\matrix"+i+".csv");
        FileReader fileReader;
        fileReader = new FileReader(file);
        CsvReader csvReader = new CsvReader(fileReader);

        while (csvReader.readRecord()){
            // 读一整行
            String temp  = csvReader.getRawRecord();
            String[] temps = temp.split(",");
            int x = Integer.parseInt(temps[0]);
            int y = Integer.parseInt(temps[1].toString().trim());
            double value = Double.parseDouble(temps[2]);
            score[x][y] = value;
//            score[y][x] = value;
        }
        csvReader.close();
    }

    private static ArrayList<ArrayList> getData(int i) throws IOException {
        File file= new File("F:\\Project\\Software Engineer\\train\\train"+i+".csv");
        FileReader fileReader;
        fileReader = new FileReader(file);
        CsvReader csvReader = new CsvReader(fileReader);
        int tempcount = 0;
        ArrayList<ArrayList> data = new ArrayList<>();
        while (csvReader.readRecord()){
            // 读一整行
            String temp  = csvReader.getRawRecord();
            String[] temps = temp.split(",");
            ArrayList list = new ArrayList();
            for (String s:temps){
                list.add(Double.parseDouble(s));
            }
            data.add(list);
            tempcount++;
        }
        csvReader.close();
        return data;
    }


}
