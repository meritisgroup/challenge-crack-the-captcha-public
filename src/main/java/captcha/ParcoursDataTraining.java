package captcha;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * @author gaeta
 *
 */
public class ParcoursDataTraining {

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

		String dataPath = trainPath;

		String modelName = "level3Conv_2";
		String resultatSuffix = ".train.err";

		MultiLayerNetwork network = MultiLayerNetwork.load(new File(modelPath + "/" + modelName + ".bin"), false);
		
		System.out.println(network.summary());
		
		System.out.println("Data directory : " + new File(dataPath).getAbsolutePath());
		File[] listFiles = new File(dataPath).listFiles();
		
		var resultat = new PrintStream(new BufferedOutputStream(new FileOutputStream(modelPath +"/"+modelName+resultatSuffix+".txt")));
		var resultatC = new PrintStream(new BufferedOutputStream(new FileOutputStream(modelPath +"/"+modelName+resultatSuffix+".stats.txt")));
		int iff = 0;
		for (File f : listFiles) {
			if ((iff++)%100 == 0) System.out.println((iff/100)+"%");
			
			//if (f.getName().contains("level1")) 
			{

				String fileName = f.getName();

				Mat originFileColor = imread(dataPath + "/" + fileName);
				
				/*
				Mat originFile = new Mat(originFileColor.size().width(), originFileColor.size().height(), COLOR_BGR2GRAY);
				Mat origin = new Mat(originFileColor.size().width(), originFileColor.size().height(), COLOR_BGR2GRAY);
				
				cvtColor(originFileColor, originFile, COLOR_BGR2GRAY);
				threshold(originFile, origin, 128, 255, THRESH_BINARY);
				
				
				imwrite(debugPath+"/originFile"+fileName+".png", originFile);
				imwrite(debugPath+"/origin"+fileName+".png", origin);
				*/
				String[] split = fileName.split("\\.");
				String name = split[0];
				
				String r = "";
				double e = 1.;
				for (int i = 0; i < 4; i++) {
					// select the region fist
					var rectCrop = new Rect(i * originFileColor.cols() / 4, 0, originFileColor.cols() / 4, originFileColor.rows());

					// generate matrix of the interested region, from original_image
//					Mat cropped = new Mat(origin, rectCrop);
//					GuessResult guess = guess(network, cropped);
//					
//					Mat croppedOrigin = new Mat(originFile, rectCrop);
//					GuessResult guessOrigin = guess(network, croppedOrigin);
//					
//					GuessResult gr = guessOrigin.confident > guess.confident ? guessOrigin: guess;
					
					Mat cropped = new Mat(originFileColor, rectCrop);
					
					var gr= guess(network, cropped);
					r+=gr.guess;
					e*=gr.confident;
				}
				if (!name.equals(r)) {
					resultat.println(name+","+r);
					resultatC.println(name+","+r+","+e);
				}
				
			}
		}
		
		System.out.println("-- Done --");
		resultat.close();
		resultatC.close();
		
	}
	

	private static GuessResult guess(MultiLayerNetwork network, Mat imageFile) throws IOException {
		int channels = 3;
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
