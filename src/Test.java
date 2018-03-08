import com.csvreader.CsvReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Administrator on 2018/3/6.
 */
public class Test {
    public static void main(String[] args) throws IOException {

        ArrayList<ArrayList> data = getData();
        System.out.println(data.size());
        System.out.println(data.get(0).size());
        LSTM lstm = new LSTM(data.size(),data.size(),data.get(0).size());
        double[][] score = Tool.getZeroMatrix(data.size(),data.size());
        updateScore(score);
        System.out.println(score[0][1]);

        int count = 0;
        while(count<20000) {
            lstm.trainForNLP(data, score);
            if (count%100==0){
                System.out.println(lstm.getLossForAll());
            }
            count++;
        }
        lstm.getResultForNLP(data);
    }

    private static void updateScore(double[][] score) throws IOException {
        File file= new File("G:\\workplace\\BLSTM\\src\\matrix1.csv");
        FileReader fileReader;
        fileReader = new FileReader(file);
        CsvReader csvReader = new CsvReader(fileReader);

        while (csvReader.readRecord()){
            // 读一整行
            String temp  = csvReader.getRawRecord();
            String[] temps = temp.split(",");
            int x = Integer.parseInt(temps[0]);
            int y = Integer.parseInt(temps[1]);
            double value = Double.parseDouble(temps[2]);
            score[x][y] = value;
        }
        csvReader.close();
    }

    private static ArrayList<ArrayList> getData() throws IOException {
        File file= new File("G:\\workplace\\BLSTM\\src\\train1.csv");
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
