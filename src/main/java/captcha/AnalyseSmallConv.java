package captcha;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class AnalyseSmallConv {

	

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder
	final static String preparePath = dataPath + "/prepare"; // Path to test folder
	final static String modelPath = dataPath + "/models"; // Path to models folder

	
	public static void main(String[] args) throws FileNotFoundException {

		String resultatSuffix = "";//".train";
		
		String modelName = "level3Conv_2";
		String coeffs = modelPath +"/"+modelName+resultatSuffix+".stats.txt";
		String modelName2 = "level3Small";
		String coeffs2 = modelPath +"/"+modelName2+resultatSuffix+".stats.txt";

		
		Map<Integer, GuessResult> datas = read(coeffs);
		Map<Integer, GuessResult> datas2 = read(coeffs2);
		
		File[] listFiles = new File(dataPath).listFiles();
		
		for (File f : listFiles) {
			
			String fileName = f.getName();
			
			if (!fileName.contains("level3")) continue;
			
			int index = Integer.parseInt(fileName.split("\\.")[0]);
			var v = datas.get(index);
			var v2 = datas2.get(index);
			
			if (!v2.guess.equals(v.guess)) {
				System.out.println(v.guess + "-"+v2.guess + " " + fileName);
			}
		}
	
	}


	private static Map<Integer, GuessResult> read(String coeffs) throws FileNotFoundException {
		Map<Integer, GuessResult> datas = new HashMap<>();
		try (Scanner scan = new Scanner(new BufferedInputStream(new FileInputStream(coeffs)))) {
			while (scan.hasNext()) {
				String line = scan.nextLine();
				var split = line.split(",");
				
				int id = Integer.parseInt(split[0]);
				String guessed = split[1];
				double confident = Double.parseDouble(split[2]);
				
				datas.put(id, new GuessResult(guessed, confident));
			}
		}
		return datas;
	}
}
