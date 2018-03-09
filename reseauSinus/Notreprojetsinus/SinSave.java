package Notreprojetsinus;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SinSave {
	/**
	 * Read a csv file. Fit and plot the data using Deeplearning4J.
	 *
	 * @author Robert Altena
	 */

	static File location = new File("Sinus.zip");
	    public static void main( String[] args ) throws IOException, InterruptedException
	    {
	    	
	        String filename = new ClassPathResource("/DataExamples/sinus.csv").getFile().getPath();
	    	DataSet ds = readCSVDataset(filename);

	    	ArrayList<DataSet> DataSetList = new ArrayList<>();
	    	DataSetList.add(ds);

	    	plotDataset(DataSetList); //Plot the data, make sure we have the right data.

	    	MultiLayerNetwork net =fitStraightline(ds);

	    	// Get the min and max x values, using Nd4j
	    	NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
	    	preProcessor.fit(ds);
	        int nSamples = 50;
	        INDArray x = Nd4j.linspace(preProcessor.getMin().getInt(0),preProcessor.getMax().getInt(0),nSamples).reshape(nSamples, 1);
	        INDArray y = net.output(x);
	        DataSet modeloutput = new DataSet(x,y);
	        DataSetList.add(modeloutput);

	    	plotDataset(DataSetList);    //Plot data and model fit.
	    	
	    	boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	        ModelSerializer.writeModel(net, location, saveUpdater);
	    }

		/**
		 * Fit a straight line using a neural network.
		 * @param ds The dataset to fit.
		 * @return The network fitted to the data
		 * @throws IOException 
		 */
	    private static DataSet readCSVDataset(String filename) throws IOException, InterruptedException{
			int batchSize = 1000;
			RecordReader rr = new CSVRecordReader();
			rr.initialize(new FileSplit(new File(filename)));

			DataSetIterator iter =  new RecordReaderDataSetIterator(rr,batchSize, 1, 1, true);
			return iter.next();
		}
	    
	    private static void plotDataset(ArrayList<DataSet> DataSetList){

			XYSeriesCollection c = new XYSeriesCollection();

			int dscounter = 1; //use to name the dataseries
			for (DataSet ds : DataSetList)
			{
				INDArray features = ds.getFeatures();
				INDArray outputs= ds.getLabels();

				int nRows = features.rows();
				XYSeries series = new XYSeries("S" + dscounter);
				for( int i=0; i<nRows; i++ ){
					series.add(features.getDouble(i), outputs.getDouble(i));
				}

				c.addSeries(series);
			}
			
			 String title = "Apprentissage Sinus";
				String xAxisLabel = "x";
				String yAxisLabel = "sin(x)";
				PlotOrientation orientation = PlotOrientation.VERTICAL;
				boolean legend = false;
				boolean tooltips = false;
				boolean urls = false;
				JFreeChart chart = ChartFactory.createScatterPlot(title , xAxisLabel, yAxisLabel, c, orientation , legend , tooltips , urls);
		    	JPanel panel = new ChartPanel(chart);

		    	 JFrame f = new JFrame();
		    	 f.add(panel);
		    	 f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		         f.pack();
		         f.setTitle("Training Data");

		         f.setVisible(true);
	    }
			
		private static MultiLayerNetwork fitStraightline(DataSet ds) throws IOException{
			
			int nEpochs = 1600;
		
		    
		    
		    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(location);
			net.init();
			net.setListeners(new ScoreIterationListener(50));
		    int listenerFrequency =1;
		    UIServer uiServer = UIServer.getInstance();
		    StatsStorage statsStorage = new InMemoryStatsStorage();
		    net.addListeners(new StatsListener(statsStorage, listenerFrequency));
		    uiServer.attach(statsStorage);
		    for( int i=0; i<nEpochs; i++ ){
		    	net.fit(ds);
		    }

		    return net;
		}
		
}