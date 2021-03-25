package captcha;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import captcha.UseNetwork.GuessResult;

/**
 * @author gaeta
 *
 */
public class ParcoursDataSolutionLevel1 {

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder
	final static String preparePath = dataPath + "/prepare"; // Path to test folder
	final static String modelPath = dataPath + "/models"; // Path to models folder


	final static String[] labels = "012345689ACDEFHKLMPQRSTUVXYZ".split("");
	
	
	public static void main(String[] args) throws IOException {
		ensureDirectories();

		System.out.println("Data directory : " + new File(testPath).getAbsolutePath());

		String modelName = "level1";

		MultiLayerNetwork network = MultiLayerNetwork.load(new File(modelPath + "/" + modelName + ".bin"), false);
		
		File[] listFiles = new File(testPath).listFiles();
		
		for (File f : listFiles) {
			if (f.getName().contains("level1")) {

				String fileName = f.getName();

				Mat origin = imread(testPath + "/" + fileName);
				
				String[] split = fileName.split("\\.");
				String name = split[0];
				
				String r = "";
				for (int i = 0; i < 4; i++) {
					// select the region fist
					var rectCrop = new Rect(i * origin.cols() / 4, 0, origin.cols() / 4, origin.rows());

					// generate matrix of the interested region, from original_image
					Mat cropped = new Mat(origin, rectCrop);
					
					GuessResult guess = guess(network, cropped);
					r+=guess.guess;
				}
				System.out.println(name+","+r);
			}
		}
		
		System.out.println("-- Done --");
		
	}
	

	private static GuessResult guess(MultiLayerNetwork network, Mat imageFile) throws IOException {
		int channels = 1;
		int width = 20;
		int height = 35;
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
	

	static void ensureDirectories() {
		ensureDirectory(debugPath);
		ensureDirectory(tmpPath);
		ensureDirectory(preparePath);
	}

	private static void ensureDirectory(String path) {
		if (!new File(path).exists())
			new File(path).mkdirs();
	}

	public static Mat bufferedImageToMat(BufferedImage bi) {
		return Java2DFrameUtils.toMat(bi);
	}

	public static BufferedImage matToBufferedImage(Mat frame) {
		return Java2DFrameUtils.toBufferedImage(frame);
	}

}
