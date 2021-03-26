package captcha;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class UseNetwork {

	final static String dataPath = "datas"; // Path to data folder
	final static String modelPath = dataPath + "/models"; // Path to models folder
	final static String trainPath = dataPath + "/gen/train"; // Path to train folder

	final static String[] labels = "OIX".split("");
	
	public static void main(String[] args) throws IOException {
		String modelName = "custom";
		int seed = 791804338;

		MultiLayerNetwork network = MultiLayerNetwork.load(new File(modelPath + "/" + modelName + ".bin"), false);
		
		Random rnd = new Random(seed);

		File parentDir = new File(trainPath);
		if (!parentDir.exists())
			throw new RuntimeException("No datas found in " + parentDir.getAbsolutePath());

		File[] dirs = parentDir.listFiles();
		File dir = dirs[rnd.nextInt(dirs.length)];
		File[] allFiles = dir.listFiles();
		File imageFile = allFiles[rnd.nextInt(allFiles.length)];

		GuessResult guess = guess(network, imageFile);

		System.out.println(guess.guess + " at " + guess.confident + " for " + dir.getName() + " (" + imageFile.getAbsolutePath() + ")");
		
	}
	
	
	private static GuessResult guess(MultiLayerNetwork network, File imageFile) throws IOException {
		int channels = 1;
		int width = 28;
		int height = 28;
		NativeImageLoader loader = new NativeImageLoader(height, width, channels);

		//put image into INDArray
		INDArray image = loader.asMatrix(imageFile);

		//values need to be scaled
		DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

		// then call that scalar on the image dataset
		scalar.transform(image);

		INDArray result = network.output(image);

        int nClasses = (int) result.data().length();
        
        double[] results = new double[nClasses];
        
        //transfer the neural network output to an array
        for (int i = 0; i < nClasses; i++) {results[i] = result.getDouble(0, i);}
        

        //display the values using helper functions defined below
        double maximum = arrayMaximum(results);
        String guess = labels[getIndexOfLargestValue(results)];
        
        return new GuessResult(guess, maximum);
	}
	
	public static class GuessResult {
		String guess;
		double confident;
		
		public GuessResult(String guess, double confident) {
			super();
			this.guess = guess;
			this.confident = confident;
		}
		
	}

	public static double arrayMaximum(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for (double cur : arr)
			max = Math.max(max, cur);
		return max;
	}

	public static int getIndexOfLargestValue(double[] array) {
		if (array == null || array.length == 0)
			return -1;
		int largest = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > array[largest])
				largest = i;
		}
		return largest;
	}

}
