/**
 * Created by Administrator on 2018/3/20.
 */
public class Pair implements Comparable{
    private int fromPos;
    private int toPos;
    private double score;

    public Pair(int fromPos, int toPos, double score) {
        this.fromPos = fromPos;
        this.toPos = toPos;
        this.score = score;
    }

    @Override
    public String toString() {
        return fromPos+", "+toPos+", "+score;
    }

    public int getFromPos() {
        return fromPos;
    }

    public void setFromPos(int fromPos) {
        this.fromPos = fromPos;
    }

    public int getToPos() {
        return toPos;
    }

    public void setToPos(int toPos) {
        this.toPos = toPos;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public int compareTo(Object o) {
        return (int) (100000*this.score-100000*((Pair) o).score);
    }
}
