package lu.kbra.pendulum_nn.swing;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Arrays;

import javax.swing.JPanel;

public class HistogramPanel extends JPanel {

	private float[] data;
	private int bins = 60;

	public void setData(float[] data) {
		this.data = data;
		repaint();
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		if (data == null || data.length == 0)
			return;

		float min = Float.MAX_VALUE;
		float max = -Float.MAX_VALUE;

		for (float v : data) {
			if (v < min)
				min = v;
			if (v > max)
				max = v;
		}

		int[] hist = new int[bins];
		for (float v : data) {
			int bin = (int) ((v - min) / (max - min) * (bins - 1));
			hist[bin]++;
		}

		int w = getWidth();
		int h = getHeight();
		int barWidth = w / bins;

		int maxCount = Arrays.stream(hist).max().orElse(1);

		g.setColor(Color.BLUE);
		for (int i = 0; i < bins; i++) {
			int barHeight = (int) ((hist[i] / (float) maxCount) * h);
			g.fillRect(i * barWidth, h - barHeight, barWidth - 1, barHeight);
		}
	}
}
