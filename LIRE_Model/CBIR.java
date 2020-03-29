import net.semanticmetadata.lire.imageanalysis.features.global.Tamura;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class CBIR {

    public double distances [];
    Tamura tams [];
    static int count=0;
    public static void main(String[] args)
    {
        try{
        int n = 120000; //upper boundary of the number of images in the database
        String paths [] = new String [n];
        Tamura   tams [] = new  Tamura  [n];
        int index [] = new int [n];
        for (int i=0; i<n; i++)
            index[i] = i;

        String path_to_dir = args[0];
        path_to_dir = "/home/yonif/SimulAI/SimulationsBW3/"; /////// DELETE THIS ROW
        FileWriter file_w = new FileWriter("/home/yonif/SimulAI/output_longtimes_try.json");
        file_w.write("{\"data\":[");
        int c= Integer.parseInt(args[1]);

        //indexing the images int the Tamura's features space
        File root = new File(path_to_dir);
        for ( File file : root.listFiles())
            IndexFilesRecursive(paths, tams, file);

        String tests =  "/home/yonif/simulai_test_longtimes";
        //String tests =  "/home/yonif/simulai_test_10000";
        root = new File(tests);
        int i = 0;
        double distances[] = new double[count];
        for (File file : root.listFiles()) {
            if(i!=0)
                file_w.write(",");
                JSONArray best_results = new JSONArray();
                CBIR_Query(distances, tams, paths, index, file, count, 2000, best_results);
                System.out.println("-------------------------DONE " + i + "----------------------------");
                i++;
                file_w.write(best_results.toJSONString());
           }
       file_w.write("]}")   ;
       file_w.close();
       }
       catch(Exception e){
           System.out.println(e.getMessage());
       }
   }

    public static void CBIR_Query (double [] distances, Tamura  [] tams, String paths [], int [] index,  File file_query, int count, int c, JSONArray best_results)
    {
        Tamura  tams_query = new  Tamura  ();
        BufferedImage img;
        try {

            img = ImageIO.read(file_query);
            tams_query.extract(img);
            for (int i = 0; i < count; i++)
                distances[i] = tams_query.getDistance(tams[i]);
            quickSort(distances, tams, paths, index, count);
            //sort(0, count - 1, distances, tams, paths, index);
            JSONObject obj;
            for (int i = 0; i < c; i++) {
                obj = new JSONObject();
                obj.put("path",  paths[i]);
                obj.put("index", index[i]);
                obj.put("lire_distance", distances[i]);
                best_results.add(obj);
                obj=null;
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void IndexFilesRecursive(String [] paths, Tamura  [] tams, File pFile)
    {
        try {
            for (File files : pFile.listFiles()) {
                if (files.isDirectory()) {
                    IndexFilesRecursive(paths, tams, files);
                } else {
                    Add_File(paths, tams, files);
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

    }

    public static void Add_File(String [] paths, Tamura   [] tams, File file)
    {
        try
        {
            paths[count] = file.getPath();
            BufferedImage img;
            img = ImageIO.read(file);
            tams[count] = new Tamura();
            tams[count].extract(img);
            count++;

            if(count %500 ==0) {
                System.out.println(count);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void swap (double [] distances, Tamura [] tams, String [] paths, int [] index, int i, int j)
    {
        double temp0 = distances[j];
        distances[j] = distances[i];
        distances[i] = temp0;
        String temp = paths[j];
        paths[j] = paths[i];
        paths[i] = temp;
        Tamura temp2 = tams[j];
        tams[j] = tams[i];
        tams[i] = temp2;
        int temp3 = index[j];
        index[j] = index[i];
        index[i] = temp3;
    }
    public static int partition(int low, int high, double [] distances, Tamura [] tams, String [] paths, int [] index)
    {
        double x = distances[low];
        int i = low-1;
        int j = high+1;
        while (true) {
            while (++i < high && distances[i] < x);
            while (--j > low && distances[j] > x);
            if (i < j) {
                swap(distances, tams, paths, index, i,j);
            } else {
                return j;
            }
        }
    }
    public static void sort(int low, int high, double [] distances, Tamura [] tams, String [] paths, int [] index)
    {
        if (low < high)
        {
            /* pi is partitioning index, arr[pi] is
              now at right place */
            int pi = partition(low, high, distances, tams, paths, index);
            // Recursively sort elements before
            // partition and after partition
            sort(low, pi, distances, tams, paths, index);
            sort(pi+1, high, distances, tams, paths, index);
        }
    }

    static final int MAX_LEVELS = 10000;
    public static boolean quickSort(double [] distances, Tamura [] tams, String [] paths, int [] index, int elements) {
        int i=0,L,R;
        double pivot;
        Tamura pivot_tamura;
        String pivot_path;
        int pivot_index;
        int[] beg = new int[MAX_LEVELS], end = new int[MAX_LEVELS];
        beg[0]=0;
        end[0]=elements;
        while(i>=0) {
            L=beg[i];
            R=end[i]-1;
            if(L<R) {
                pivot=distances[L];
                pivot_tamura = tams[L];
                pivot_index = index[L];
                pivot_path = paths[L];
                if(i==MAX_LEVELS-1) return false;
                while(L<R) {
                    while(distances[R]>=pivot&&L<R)
                        R--;
                    if(L<R) {
                        distances[L] = distances[R];
                        tams[L] = tams[R];
                        index[L] = index[R];
                        paths[L] = paths[R];
                        L++;
                    }
                    while(distances[L]<=pivot&&L<R)
                        L++;
                    if(L<R)
                    {
                        distances[R]=distances[L];
                        tams[R] = tams[L];
                        index[R] = index[L];
                        paths[R] = paths[L];
                        R--;
                    }
                }
                distances[L]=pivot;
                tams[L] = pivot_tamura;
                index[L] = pivot_index;
                paths[L] = pivot_path;
                beg[i+1]=L+1;
                end[i+1]=end[i];
                end[i++]=L;
            } else {
                i--;
            }
        }
        return true;
    }


}
