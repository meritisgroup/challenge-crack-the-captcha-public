package captcha;

import java.io.File;
import java.util.Scanner;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;

public class ViewNetworkLearningData {

	final static String dataPath = "datas"; // Path to data folder
	final static String modelPath = dataPath + "/models"; // Path to models folder
	final static String logsPath = dataPath + "/models/logs"; // Path to models folder


	
	public static void main(String[] args) {
		String statsFileName = "statsLevel1";
		
		StatsStorage statsStorage = new FileStatsStorage(new File(logsPath + "/"+statsFileName+".bin"));
		// Listener for an UI on http://localhost:9000
		UIServer uiServer = UIServer.getInstance();
		uiServer.attach(statsStorage);
		
		
		// wait before stopping
		try (var scan = new Scanner(System.in)) {scan.nextLine();}
	}
	
}
