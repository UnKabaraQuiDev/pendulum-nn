package lu.kbra.pendulum_nn;

import java.awt.BorderLayout;
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import lu.pcy113.pclib.datastructure.pair.Pair;
import lu.pcy113.pclib.swing.JLineGraph;
import lu.pcy113.pclib.swing.JLineGraph.ChartData;
import lu.pcy113.pclib.swing.JLineGraph.RangeChartData;

public class NNFrame extends JFrame {

	private HistogramPanel histogramPanel;
	private JLineGraph history;

	private List<Double> avgHistory = new ArrayList<>();
	private List<Double> maxHistory = new ArrayList<>();
	private List<Double> minHistory = new ArrayList<>();
	private List<Pair<Double, Double>> stdDevHistory = new ArrayList<>();

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
		stdDevCd.setValues(stdDevHistory);
		ChartData avgCd = history.createSeries("Average");
		avgCd.setBorderColor(Color.RED);
		avgCd.setValues(avgHistory);
		ChartData maxCd = history.createSeries("Max");
		maxCd.setBorderColor(Color.BLUE);
		maxCd.setValues(maxHistory);
		ChartData minCd = history.createSeries("Min");
		minCd.setBorderColor(Color.GREEN);
		minCd.setValues(minHistory);
	}

	public HistogramPanel getHistogramPanel() {
		return histogramPanel;
	}

	public JLineGraph getHistory() {
		return history;
	}

	public List<Double> getAvgHistory() {
		return avgHistory;
	}

	public List<Double> getMinHistory() {
		return minHistory;
	}

	public List<Double> getMaxHistory() {
		return maxHistory;
	}

	public List<Pair<Double, Double>> getStdDevHistory() {
		return stdDevHistory;
	}
}
