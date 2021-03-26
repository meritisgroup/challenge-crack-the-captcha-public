package captcha;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.function.Predicate;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

public class ParcoursData {

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder
	final static String preparePath = dataPath + "/prepare"; // Path to test folder

	public static void main(String[] args) {
		mainF(preparePath, __ -> true);
	}
	public static void mainF(String path, Predicate<String> filter) {
		ensureDirectories(path);

		System.out.println("Data directory : " + new File(dataPath).getAbsolutePath());

		int compteur = 0;
		
		
		File[] listFiles = new File(trainPath).listFiles();
		
		for (File f : listFiles) {
			if (filter.test(f.getName())) 
			{

				String fileName = f.getName();

				Mat origin = imread(trainPath + "/" + fileName);
				imwrite(debugPath + "/origin.png", origin);

				String[] split = fileName.split("\\.");
				String[] letters = split[0].split("");
				for (int i = 0; i < 4; i++) {
					// select the region fist
					var rectCrop = new Rect(i * origin.cols() / 4, 0, origin.cols() / 4, origin.rows());

					// generate matrix of the interested region, from original_image
					Mat cropped = new Mat(origin, rectCrop);
					//Mat gray = new Mat(cropped.size().width(), cropped.size().height(), COLOR_BGR2GRAY);

					//cvtColor(cropped, gray, COLOR_BGR2GRAY);
					
					String letterPath = path + "/" + letters[i];
					ensureDirectory(letterPath);

					imwrite(letterPath + "/" + (compteur++) + ".png", cropped);
					//imwrite(letterPath + "/" + (compteur++) + ".png", gray);
				}
			}
		}
		
		System.out.println("-- Done --");
		
	}
	
	
	

	static void ensureDirectories(String path) {
		ensureDirectory(debugPath);
		ensureDirectory(tmpPath);
		ensureDirectory(path);
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
