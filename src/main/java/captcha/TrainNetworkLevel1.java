package captcha;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainNetworkLevel1 {

	final static String dataPath = "datas"; // Path to data folder
	final static String modelPath = dataPath + "/models"; // Path to models folder
	final static String logsPath = dataPath + "/models/logs"; // Path to models folder
	final static String trainPath = dataPath + "/prepare"; // Path to train folder

	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final Random randNumGen = new Random(735122); // Allow replay

	public static void main(String[] args) throws IOException {
		int batchSize = 2048; // how many examples to simultaneously train in the network
		int rngSeed = 3289322;
		int height = 35;
		int width = 20;
		int channels = 1;
		int numEpochs = 40;

		String modelName = "level1";
		String statsFileName = "statsLevel1";
		
		ensureDirectory(modelPath);
		ensureDirectory(logsPath);

		File parentDir = new File(trainPath);
		if (!parentDir.exists())
			throw new RuntimeException("No datas found in " + parentDir.getAbsolutePath());

		int classes = parentDir.list().length;
		var conf = networkConfiguration(height, width, 1, classes, rngSeed);
		var network = new MultiLayerNetwork(conf);

		network.init();

		System.out.println(network.summary());

		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20); // 80% train 20% tests
		InputSplit trainData = filesInDirSplit[0];
		InputSplit testData = filesInDirSplit[0];

		ImageTransform transform = new MultiImageTransform();
		//ImageTransform transform = new MultiImageTransform(randNumGen, new RotateImageTransform(10.f));

		// Normalize entre 0 et 1
		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler();

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordReader.initialize(trainData, transform);

		ImageRecordReader recordTestReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordTestReader.initialize(testData);

		int outputNum = recordReader.numLabels();

		int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by
							// recordReader.next()

		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
		dataIter.setPreProcessor(imagePreProcessingScaler);

		DataSetIterator dataTestIter = new RecordReaderDataSetIterator(recordTestReader, batchSize, labelIndex,
				outputNum);
		dataTestIter.setPreProcessor(imagePreProcessingScaler);

		// pass a training listener that reports score every 10 iterations
		int listenerFrequency = 1;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = true;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));

		StatsStorage statsStorage = new FileStatsStorage(new File(logsPath + "/"+statsFileName+".bin"));
		network.addListeners(new StatsListener(statsStorage, listenerFrequency));
		// Listener for an UI on http://localhost:9000
		// UIServer uiServer = UIServer.getInstance();
		// uiServer.attach(statsStorage);

		System.out.println(
				"Training workspace config: " + network.getLayerWiseConfigurations().getTrainingWorkspaceMode());
		System.out.println(
				"Inference workspace config: " + network.getLayerWiseConfigurations().getInferenceWorkspaceMode());

		// simply use for loop
		for (int i = 0; i < numEpochs; i++) {
			System.out.println("Epoch " + i + " / " + numEpochs);
			long start = System.currentTimeMillis();
			network.fit(dataIter);
			long end = System.currentTimeMillis();
			System.out.println("Epoch " + i + " / " + numEpochs + " -> " + ((end - start) / 1000) + "s");
		}
		network.save(new File(modelPath+"/"+ modelName + ".bin"));

		// evaluate basic performance
		var eval = network.evaluate(dataTestIter);
		System.out.print(eval.stats(false, true));

		System.out.println(" ------   end   ------");
	}

	private static MultiLayerConfiguration networkConfiguration(int numRows, int numColumns, int channels,
			int outputNum, int rngSeed) {
		return new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.cacheMode(CacheMode.HOST)
				.updater(new Nesterovs(0.005, 0.9)) // learning rate, momentum
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list()
				.layer(new DenseLayer.Builder()
						.name("dense1")
						.activation(Activation.RELU)
						.nOut(256).build())
				/*.layer(new DenseLayer.Builder()
						.name("dense2")
						.activation(Activation.RELU)
						.nOut(128).build())*/
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.name("output")
						.activation(Activation.SOFTMAX)
						.nOut(outputNum).build())
				.setInputType(InputType.convolutional(numRows, numColumns, channels)) // InputType.convolutional for normal image or convolutionalFlat 
				//.setInputType(InputType.convolutional(numRows, numColumns, channels)) // InputType.convolutional for normal image or convolutionalFlat 
				.build();
	}
	
	private static void ensureDirectory(String path) {
		if (!new File(path).exists())
			new File(path).mkdirs();
	}
}
