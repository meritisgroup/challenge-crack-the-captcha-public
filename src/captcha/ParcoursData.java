package captcha;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.awt.image.BufferedImage;
import java.io.File;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

public class ParcoursData {

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder

	public static void main(String[] args) {
		ensureDirectories();

		System.out.println("Data directory : " + new File(dataPath).getAbsolutePath());

		String fileName = "0EQP.level1.png";

		Mat origin = imread(trainPath + "/" + fileName);
		imwrite(debugPath + "/origin.png", origin);

		// select the region fist
		var rectCrop = new Rect(0, 0, origin.cols() / 4, origin.rows());

		// generate matrix of the interested region, from original_image
		Mat cropped = new Mat(origin, rectCrop);

		imwrite(debugPath + "/letter0.png", cropped);

	}

	static void ensureDirectories() {
		ensureDirectory(debugPath);
		ensureDirectory(tmpPath);
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
