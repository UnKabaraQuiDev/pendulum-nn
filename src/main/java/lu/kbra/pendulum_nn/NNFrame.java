package lu.kbra.pendulum_nn;

import java.awt.BorderLayout;
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.apache.commons.collections4.queue.CircularFifoQueue;

import lu.pcy113.pclib.datastructure.pair.Pair;
import lu.pcy113.pclib.swing.JLineGraph;
import lu.pcy113.pclib.swing.JLineGraph.ChartData;
import lu.pcy113.pclib.swing.JLineGraph.RangeChartData;

public class NNFrame extends JFrame {

	private HistogramPanel histogramPanel;
	private JLineGraph history;

	private CircularFifoQueue<Double> avgHistory = new CircularFifoQueue<>(500);
	private CircularFifoQueue<Double> maxHistory = new CircularFifoQueue<>(500);
	private CircularFifoQueue<Double> minHistory = new CircularFifoQueue<>(500);
	private CircularFifoQueue<Pair<Double, Double>> stdDevHistory = new CircularFifoQueue<>(500);

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
		history.setNextBorderWidth(1);
//		history.overrideMaxValue(PALogic.VIRTUAL_SECONDS);
		RangeChartData stdDevCd = history.createRangeSeries("StdDev");
		stdDevCd.setBorderColor(Color.YELLOW);
		stdDevCd.setFillColor(new Color(Color.YELLOW.getRed(), Color.YELLOW.getGreen(), Color.YELLOW.getBlue(), 100));
		stdDevCd.setFill(true);
		stdDevCd.setValues(stdDevHistory, i -> stdDevHistory.get(i).getKey(), i -> stdDevHistory.get(i).getValue());
		ChartData avgCd = history.createSeries("Average");
		avgCd.setBorderColor(Color.RED);
		avgCd.setValues(avgHistory, i -> avgHistory.get(i));
		ChartData maxCd = history.createSeries("Max");
		maxCd.setBorderColor(Color.BLUE);
		maxCd.setValues(maxHistory, i -> maxHistory.get(i));
		ChartData minCd = history.createSeries("Min");
		minCd.setBorderColor(Color.GREEN);
		minCd.setValues(minHistory, i -> minHistory.get(i));
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
}
