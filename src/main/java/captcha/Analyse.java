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

public class Analyse {

	

	final static String dataPath = "datas"; // Path to data folder
	final static String debugPath = dataPath + "/debug"; // Path to debug folder
	final static String tmpPath = dataPath + "/tmp"; // Path to tmp folder
	final static String trainPath = dataPath + "/train"; // Path to train folder
	final static String testPath = dataPath + "/test"; // Path to test folder
	final static String preparePath = dataPath + "/prepare"; // Path to test folder
	final static String modelPath = dataPath + "/models"; // Path to models folder

	
	public static void main(String[] args) throws FileNotFoundException {

		String modelName = "level3Bg_2_";
		String resultatSuffix = ".train";
		
		String coeffs = modelPath +"/"+modelName+resultatSuffix+".stats.txt";

		String output = modelPath +"/"+modelName+".complet.txt";
		var resultat = new PrintStream(new BufferedOutputStream(new FileOutputStream(output)));
		
		
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
		Map<Integer, String> datasHelper = new HashMap<>();
		try (Scanner scan = new Scanner(new BufferedInputStream(new FileInputStream(dataPath+"/helper.txt")))) {
			while (scan.hasNext()) {
				String line = scan.nextLine();
				var split = line.split(",");
				
				int id = Integer.parseInt(split[0]);
				String guessed = split[1];
				
				datas.put(id, new GuessResult(guessed, 1.));
				datasHelper.put(id,  guessed);
			}
		}
		
		datas.entrySet().stream().forEach(e -> {
			resultat.println(e.getKey() +","+e.getValue().guess);
		});
		
		resultat.close();
	}
}
