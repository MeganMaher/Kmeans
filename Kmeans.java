import java.io.*;
import java.util.ArrayList;

public class Kmeans{

	private final double TOLERANCE=.01;

    public double[][] finalCentroids;

	public static void main(String[] args){
		Kmeans km=new Kmeans();
        int k = 4;
		double[][] inst=km.read("proj02data.csv");

        // Normalize data
        inst = km.normalizeData(inst);

        int[] c=km.cluster(inst,k);

        System.out.println("\n--- Centroids ---");
        km.printMatrix(km.finalCentroids);

        km.printClusters(c, k);

        km.printDescription(km.finalCentroids, k, inst, c);

        km.printOutliers(inst, c, km.finalCentroids, k);
    }

    public double[][] normalizeData(double[][] inst) {

        double[] max = new double[inst[0].length];
        double[] min = new double[inst[0].length];

        /* Initialize min array */
        for (int i = 0; i < min.length; i++) {
            min[i] = Double.MAX_VALUE;
        }

        /* Get max and mins */
        for (int r = 0; r < inst.length; r++) {
            for (int c = 0; c < inst[r].length; c++) {
                double current = inst[r][c];
                if (current > max[c]) {
                    max[c] = current;
                }

                if (current < min[c]) {
                    min[c] = current;
                }
            }
        }

        for (int r = 0; r < inst.length; r++) {
            for (int c = 0; c < inst[r].length; c++) {
                inst[r][c] = (inst[r][c] - min[c])/(max[c] - min[c]);
            }
        }

        return inst;
    }

    public void printOutliers(double[][] inst, int[] cluster, double[][] centroids, int k) {
        System.out.println("\n---- Outliers ----");
        for (int j = 0; j < k; j++) {
            int fartherst_instance = -1;
            double farthest_value = 0;

            for (int i = 0; i < cluster.length; i++) {
                if (cluster[i] != j) continue;

                double dist = euclid(inst[i], centroids[j]);

                if (dist > farthest_value) {
                    farthest_value = dist;
                    fartherst_instance = i;
                }
            }
            if (fartherst_instance == -1) {
                System.out.println(j + ": <No Outlier>");

            } else {
                System.out.println(j + ": Instance " + fartherst_instance + ", distance " + farthest_value);
            }
        }
    }

	public int[] cluster(double[][] inst, int k){
		int[] clusters=new int[inst.length];
		double[][] centroids=init(inst,k);
		double errThis=sse(inst,centroids,clusters), errLast=errThis+1;
		while(errLast-errThis>TOLERANCE){
			//reassign the clusters using assignClusters
            clusters = assignClusters(inst, centroids);
            //re-calculate the centroids
            centroids = recalcCentroids(inst, clusters, k);
            //re-calculate the error using sse
            errLast = errThis;
            errThis = sse(inst, centroids, clusters);
        }

        finalCentroids = centroids;
		return clusters;
	}

	//finds initial clusters - no modifications necessary
	public double[][] init(double[][] inst, int k){
		int n=inst.length, d=inst[0].length;
		double[][] centroids=new double[k][d];
		double[][] extremes=new double[d][2];
		for(int i=0; i<d; i++)
			extremes[i][1]=Double.MAX_VALUE;
		for(int i=0; i<n; i++)
			for(int j=0; j<d; j++){
				extremes[j][0]=Math.max(extremes[j][0],inst[i][j]);
				extremes[j][1]=Math.min(extremes[j][1],inst[i][j]);
			}
		for(int i=0; i<k; i++)
			for(int j=0; j<d; j++)
				centroids[i][j]=Math.random()*(extremes[j][0]-extremes[j][1])+extremes[j][1];
		return centroids;
	}

	public int[] assignClusters(double[][] inst, double[][] centroids){
		int n=inst.length, d=inst[0].length, k=centroids.length;
		int[] rtn=new int[n];
		//for each instance
		for(int i = 0; i < n; i++) {
            		//calculate the distance to each of the different centroids
            		double min_distance = Double.MAX_VALUE;
            		int closest_centroid = 0;
            		for (int c = 0; c < k; c++) {
                		double dist = euclid(inst[i], centroids[c]);
                		if (dist < min_distance) {
                   			min_distance = dist;
                   		 	closest_centroid = c;
               			 }
            		}

           		 //and assign it to the cluster with the lowest distance
		    	rtn[i] = closest_centroid;
       		 }

        return rtn;
	}


	public double[][] recalcCentroids(double[][] inst, int[] clusters, int k){
		int n=inst.length, d=inst[0].length;
//		double[][] centroids=new double[k][d];
		int[] cnt=new int[k];

        double[][] sums = new double[k][d];
		//use cnt to count the number of instances in each cluster
		//for each cluster
        for (int c = 0; c < k; c++) {
            //for each attribute in this cluster
            for (int i = 0; i < n; i++) {
                if (clusters[i] != c) continue;
                cnt[c] ++;
                //add the value of the attribute from each instance in the cluster
                for (int a = 0; a < d; a++) {
                    sums[c][a] += inst[i][a];
                }
            }
        }

        //calculate the averages by dividing each attribute total by the count
        double[][] avg = new double[k][d];

        //do this for each centroid, each attribute
        for (int c = 0; c < k; c++) {
            //be careful not to divide by zero - if a cluster is emply, skip it
            if (cnt[c] == 0) continue;

            for (int a = 0; a < d; a++) {
                avg[c][a] = (double) sums[c][a] / (double) cnt[c];
            }
        }

		return avg;
	}

	public double sse(double[][] inst, double[][] centroids, int[] clusters){
		int n=inst.length, d=inst[0].length;
		double sum=0;
		//iterate through all instances
		for (int ir = 0; ir < n; ir++){
            for (int ic = 0; ic < d; ic++) {
                //iterate through all clusters
                for (int c = 0; c < clusters.length; c++) {
                    //if an instance is in the current cluster...
                    if (clusters[ir] != c) continue;

                    // add the euclidean distance between them to the sum
                    sum += euclid(inst[ir], centroids[c]);
                }
            }
        }


		return sum;
	}

	private double euclid(double[] inst1, double[] inst2){
		double sum=0;
		//calculate the euclidean distance between inst1 and inst2
        for (int i = 0; i < inst1.length; i++) {
            sum += Math.pow(inst1[i] - inst2[i], 2);
        }

		return Math.sqrt(sum);
	}

	//prints out a matrix - can be used for debugging - no modifications necessary
	public void printMatrix(double[][] mat){
		for(int i=0; i<mat.length; i++){
			for(int j=0; j<mat[i].length; j++)
				System.out.print(mat[i][j]+"\t");
			System.out.println();
		}
	}

    public void printMatrix(int[][] mat){
        for(int i=0; i<mat.length; i++){
            for(int j=0; j<mat[i].length; j++)
                System.out.print(mat[i][j]+"\t");
            System.out.println();
        }
    }

    public void printClusters(int[] clusters, int k) {
        System.out.println("\n---- Clusters ----");
        for (int j = 0; j < k; j++) {
            System.out.print(j + ": [");

            if (clusters.length > 0){
                int i;
                for (i = 0; i < clusters.length; i++) {
                    if (clusters[i] == j) {
                        System.out.print(i + ", ");
                    }
                }
                System.out.println("]");
            } else {
                System.out.println(" ]");
            }
        }
    }

    public void printDescription(double [][]centroids, int k, double[][] init, int[] clusters) {

        System.out.println("\n---- Description ----");
        double[] averages = new double[init[0].length];

        for (int c = 0; c < init[0].length; c++) {
            double sum = 0;

            for (int r = 0; r < init.length; r++) {
                sum += init[r][c];
            }

            averages[c] = sum / init.length;
        }


        for (int c = 0; c < k; c++) {
            System.out.println("Cluster " + c + ": ");

            if (centroids[c][0] == 0) {
                System.out.println("\t-- Empty Cluster --");
                continue;
            }

            for (int col = 0; col < centroids[0].length; col++) {

                String columnName = getColName(col);

                System.out.print("\t" + columnName + ": \t");
                double val = centroids[c][col];
                double avg = averages[col];

//                System.out.print(val + "; \t" + "Avg: \t" + avg + ", Diff: \t" + (val - avg));
                System.out.printf("%05f; \tAvg: %05f, \tDiff: %05f", val, avg, (val-avg));
                System.out.println();
            }

            System.out.println();
            System.out.print("\t");
            for (int col = 0; col < centroids[0].length; col++) {
                double val = centroids[c][col];
                double avg = averages[col];
                double diff = val - avg;
                double thresh = .3 * avg;
                String columnName = getColName(col);

                if (diff > thresh) {
                    System.out.print("High " + columnName + " ");
                } else if (diff < thresh) {
                    System.out.print("Low " + columnName + " ");
                }
            }
            System.out.println();
        }
    }

    private String getColName(int colNum) {
        String columnName = "";
        switch (colNum) {
            case 0:
                columnName = "Age   ";
                break;
            case 1:
                columnName = "Height";
                break;
            case 2:
                columnName = "Weight";
                break;
            case 3:
                columnName = "Dash  ";
                break;
            case 4:
                columnName = "Bench ";
                break;
            default: return "";
        }

        return columnName;
    }

	//reads in the file - no modifications necessary
	public double[][] read(String filename){
		double[][] rtn=null;
		try{
			BufferedReader br=new BufferedReader(new FileReader(filename));
			ArrayList<String> lst=new ArrayList<String>();
			br.readLine();//skip first line of file - headers
			String line="";
			while((line=br.readLine())!=null)
				lst.add(line);
			int n=lst.size(), d=lst.get(0).split(",").length;
			rtn=new double[n][d];
			for(int i=0; i<n; i++){
				String[] parts=lst.get(i).split(",");
				for(int j=0; j<d; j++)
					rtn[i][j]=Double.parseDouble(parts[j]);
			}
			br.close();
		}catch(IOException e){System.out.println(e.toString());}
		return rtn;
	}
}
