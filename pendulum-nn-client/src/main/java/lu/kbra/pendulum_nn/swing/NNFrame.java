package lu.kbra.pendulum_nn.swing;

import java.awt.BorderLayout;
import java.awt.Color;

import javax.swing.JFrame;

import org.apache.commons.collections4.queue.CircularFifoQueue;

import lu.kbra.pclib.datastructure.pair.Pair;
import lu.kbra.pclib.swing.JLineGraph;
import lu.kbra.pclib.swing.JLineGraph.ChartData;
import lu.kbra.pclib.swing.JLineGraph.RangeChartData;

public class NNFrame extends JFrame {

	private HistogramPanel histogramPanel;
	private JLineGraph history;

	public static final int BUFFER_LENGTH = 500;

	private final CircularFifoQueue<Double> avgHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> maxHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> minHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Pair<Double, Double>> stdDevHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> startingPosHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> startingAngleHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> startingVelHistory = new CircularFifoQueue<>(BUFFER_LENGTH);
	private final CircularFifoQueue<Double> startingAngleVelHistory = new CircularFifoQueue<>(BUFFER_LENGTH);

	public NNFrame() {
		super("...");

		super.setSize(400, 300);
		super.setVisible(true);
		super.setLayout(new BorderLayout());
		super.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		super.add(history = new JLineGraph(), BorderLayout.CENTER);
		super.add(history.createLegend(false, false), BorderLayout.SOUTH);
		history.setNextFilled(false);
//		history.setBackground(Color.);
		history.setMinorAxisStep(20);
		history.setNextBorderWidth(2);
//		history.overrideMaxValue(PALogic.VIRTUAL_SECONDS);
		final RangeChartData stdDevCd = history.createRangeSeries("StdDev");
		stdDevCd.setBorderColor(Color.YELLOW);
		stdDevCd.setFillColor(new Color(Color.YELLOW.getRed(), Color.YELLOW.getGreen(), Color.YELLOW.getBlue(), 100));
		stdDevCd.setFill(true);
		stdDevCd.setValues(stdDevHistory, i -> stdDevHistory.get(i).getKey(), i -> stdDevHistory.get(i).getValue());
		final ChartData avgCd = history.createSeries("Average");
		avgCd.setBorderColor(Color.RED);
		avgCd.setValues(avgHistory, avgHistory::get);
		final ChartData maxCd = history.createSeries("Max");
		maxCd.setBorderColor(Color.BLUE);
		maxCd.setValues(maxHistory, maxHistory::get);
		final ChartData minCd = history.createSeries("Min");
		minCd.setBorderColor(Color.GREEN);
		minCd.setValues(minHistory, minHistory::get);

		final ChartData startingPosCd = history.createSeries("Starting position");
		startingPosCd.setBorderColor(Color.ORANGE);
		startingPosCd.setValues(startingPosHistory, startingPosHistory::get);
		final ChartData startingVelCd = history.createSeries("Starting velocity");
		startingVelCd.setBorderColor(Color.PINK);
		startingVelCd.setValues(startingVelHistory, startingVelHistory::get);
		final ChartData startingAngleCd = history.createSeries("Starting angle");
		startingAngleCd.setBorderColor(Color.MAGENTA);
		startingAngleCd.setValues(startingAngleHistory, startingAngleHistory::get);
		final ChartData startingAngleVelCd = history.createSeries("Starting angle velocity");
		startingAngleVelCd.setBorderColor(Color.CYAN);
		startingAngleVelCd.setValues(startingAngleVelHistory, startingAngleVelHistory::get);
	}

	public HistogramPanel getHistogramPanel() {
		return histogramPanel;
	}

	public JLineGraph getHistory() {
		return history;
	}

	public CircularFifoQueue<Double> getAvgHistory() {
		return avgHistory;
	}

	public CircularFifoQueue<Double> getMinHistory() {
		return minHistory;
	}

	public CircularFifoQueue<Double> getMaxHistory() {
		return maxHistory;
	}

	public CircularFifoQueue<Pair<Double, Double>> getStdDevHistory() {
		return stdDevHistory;
	}

	public CircularFifoQueue<Double> getStartingPosHistory() {
		return startingPosHistory;
	}

	public CircularFifoQueue<Double> getStartingAngleHistory() {
		return startingAngleHistory;
	}

	public CircularFifoQueue<Double> getStartingVelHistory() {
		return startingVelHistory;
	}

	public CircularFifoQueue<Double> getStartingAngleVelHistory() {
		return startingAngleVelHistory;
	}

}
