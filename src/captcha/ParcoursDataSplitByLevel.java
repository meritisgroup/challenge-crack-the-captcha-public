package captcha;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;


public class ParcoursDataSplitByLevel {

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder
	final static String levelPath = dataPath + "/level"; // Path to train folder

	public static void main(String[] args) throws IOException {
		ensureDirectories();

		System.out.println("Data directory : " + new File(dataPath).getAbsolutePath());
		
		for (File f : new File(trainPath).listFiles()) {
			String[] split = f.getName().split("\\.");
			String name = split[0];
			String level = split[1];
			String ext = split[2];
			
			File dest = new File(levelPath, level);
			if (!dest.exists()) dest.mkdirs();
			
			Files.copy(f.toPath(), new File(levelPath + "/" + level + "/" + name +"."+ext ).toPath());
		}

	}

	static void ensureDirectories() {
		ensureDirectory(levelPath);
	}

	private static void ensureDirectory(String path) {
		if (!new File(path).exists())
			new File(path).mkdirs();
	}

}
