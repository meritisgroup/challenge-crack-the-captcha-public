package captcha;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.deeplearning4j.ui.model.stats.StatsListener;
//import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.github.marschall.memoryfilesystem.MemoryFileSystemBuilder;


public class TrainNetworkLevel3Conv2 {

	private static final class MemFSImageRecordReader extends ImageRecordReader {
		private final Path samplesPath;
		private static final long serialVersionUID = 1L;

		private MemFSImageRecordReader(long height, long width, long channels, PathLabelGenerator labelGenerator,
				Path samplesPath) {
			super(height, width, channels, labelGenerator);
			this.samplesPath = samplesPath;
		}

		@Override
		    public List<Writable> next() {
		        if (iter != null) {
		            List<Writable> ret;
		            File image = iter.next();
		            currentFile = image;

		            if (image.isDirectory())
		                return next();
		            try {
		                invokeListeners(image);
		                
		                INDArray array = imageLoader.asMatrix(
		                		Files.newInputStream(samplesPath.resolve(image.getParentFile().getName()).resolve(image.getName())));
		                //INDArray array = imageLoader.asMatrix(image);
		                if(!nchw_channels_first){
		                    array = array.permute(0,2,3,1);     //NCHW to NHWC
		                }

		                Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.DEVICE);
		                ret = RecordConverter.toRecord(array);
		                if (appendLabel || writeLabel){
		                    if(labelMultiGenerator != null){
		                        ret.addAll(labelMultiGenerator.getLabels(image.getPath()));
		                    } else {
		                        if (labelGenerator.inferLabelClasses()) {
		                            //Standard classification use case (i.e., handle String -> integer conversion
		                            ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath()))));
		                        } else {
		                            //Regression use cases, and PathLabelGenerator instances that already map to integers
		                            ret.add(labelGenerator.getLabelForPath(image.getPath()));
		                        }
		                    }
		                }
		            } catch (Exception e) {
		                throw new RuntimeException(e);
		            }
		            return ret;
		        } else if (record != null) {
		            hitImage = true;
		            invokeListeners(record);
		            return record;
		        }
		        throw new IllegalStateException("No more elements");
		    }
	}


	final static String dataPath = "datas"; // Path to data folder
	final static String modelPath = dataPath + "/models"; // Path to models folder
	final static String logsPath = dataPath + "/models/logs"; // Path to models folder
	final static String trainPath = dataPath + "/prepare"; // Path to train folder

	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final Random randNumGen = new Random(735122); // Allow replay

	public static void main(String[] args) throws IOException {
		TrainNetworkLevel3Conv2.mainN(args);
		ParcoursDataSolutionLevel1.mainR("level3Bg_2_");
	}
	public static void mainN(String[] args) throws IOException {
		int batchSize = 1024; // how many examples to simultaneously train in the network
		int rngSeed = 3289322;
		int height = 35;
		int width = 20;
		int channels = 3;
		int numEpochs = 300;

		String modelName = "level3Bg_2_";
		String statsFileName = "stats_"+modelName;
		
		ensureDirectory(modelPath);
		ensureDirectory(logsPath);

		File parentDir = new File(trainPath);
		if (!parentDir.exists())
			throw new RuntimeException("No datas found in " + parentDir.getAbsolutePath());

		int classes = parentDir.list().length;
		var conf = networkConfiguration(height, width, channels, classes, rngSeed);
		var network = new MultiLayerNetwork(conf);

		network.init();

		System.out.println(network.summary());

		
		
		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter); // 80% train 20% tests
		InputSplit[] filesInDirSplitTest = filesInDir.sample(pathFilter); // 80% train 20% tests
		InputSplit trainData = filesInDirSplit[0];
		InputSplit testData = filesInDirSplitTest[0];
		
		

		//ImageTransform transform = null;
		ImageTransform transform = new MultiImageTransform(randNumGen
				, new ScaleImageTransform(randNumGen, 4.f)
				, new RotateImageTransform(randNumGen, 10.f));

		// Normalize entre 0 et 1
		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler();

		Path samplesPath = copyIntoMemory(trainPath);
		ImageRecordReader recordReader = new MemFSImageRecordReader(height, width, channels, labelMaker, samplesPath);
		recordReader.initialize(trainData, transform);

		ImageRecordReader recordTestReader = new MemFSImageRecordReader(height, width, channels, labelMaker, samplesPath);
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
		int listenerFrequency = 1000;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = false;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));

		//StatsStorage statsStorage = new FileStatsStorage(new File(logsPath + "/"+statsFileName+".bin"));
		//network.addListeners(new StatsListener(statsStorage, listenerFrequency));
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
				.cacheMode(CacheMode.DEVICE)
				//.updater(new Nesterovs(0.001, 0.9)) // learning rate, momentum
				.updater(new Nesterovs(0.0005, 0.9)) // learning rate, momentum
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.dropOut(0.95)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)//5, 5
						.name("filter")
						.nIn(channels)
						.stride(1, 1)
						.nOut(16)  //20
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)//5, 5
						.name("conv1")
						.stride(1, 1)
						.nOut(32)  //20
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool1")
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.name("conv2")
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(64)
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool2")
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.name("conv3")
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool3")
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.name("dense1")
						.activation(Activation.RELU)
						.nOut(512).build())
				.layer(new DenseLayer.Builder()
						.name("dense2")
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
				.setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image or convolutionalFlat 
				.build();
	}
	
	private static void ensureDirectory(String path) {
		if (!new File(path).exists())
			new File(path).mkdirs();
	}
	
	public static Path copyIntoMemory(String path) throws IOException {
		File parentDir = new File(path);
		Path samplesPath = parentDir.toPath();
		FileSystem fileSystem = MemoryFileSystemBuilder.newEmpty().build();
		Path p = fileSystem.getPath("p");
		Files.createDirectories(p);
		copyFolder(samplesPath, p);
		
		return p;
	}
	
	
	public static void copyFolder(Path src, Path dest) {
	    try {
	        Files.walk( src ).forEach( s -> {
	            try {
	                Path d = dest.resolve( src.relativize(s).toString().replace('\\', '/') );
	                if( Files.isDirectory( s ) ) {
	                    if( !Files.exists( d ) )
	                        Files.createDirectory( d );
	                    return;
	                }
	                Files.copy( s, d );// use flag to override existing
	            } catch( Exception e ) {
	                e.printStackTrace();
	            }
	        });
	    } catch( Exception ex ) {
	        ex.printStackTrace();
	    }
	}
}
