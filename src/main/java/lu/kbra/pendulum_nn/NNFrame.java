package lu.kbra.pendulum_nn;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class NNFrame extends JFrame {

	public static class NNPanel extends JPanel {

		public static final float WEIGHT_SCALE = 5;

		private final NNInstance nn;

		public NNPanel(NNInstance nn) {
			this.nn = nn;
		}

		@Override
		protected void paintComponent(Graphics g) {
			System.err.println("drawing");
			super.paintComponent(g);
			Graphics2D g2 = (Graphics2D) g;

			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

			NNStructure s = nn.structure;

			int[] layers = buildLayers(s);
			Point[][] neurons = layoutNeurons(layers);

			drawConnections(g2, layers, neurons);
			drawNeurons(g2, layers, neurons);
		}

		private int[] buildLayers(NNStructure s) {
			int[] layers = new int[2 + s.innerLayers.length];
			layers[0] = s.inputCount;
			System.arraycopy(s.innerLayers, 0, layers, 1, s.innerLayers.length);
			layers[layers.length - 1] = s.outputCount;
			return layers;
		}

		private Point[][] layoutNeurons(int[] layers) {
			int w = getWidth();
			int h = getHeight();

			Point[][] pts = new Point[layers.length][];

			for (int l = 0; l < layers.length; l++) {
				pts[l] = new Point[layers[l]];

				int x = (l + 1) * w / (layers.length + 1);
				for (int i = 0; i < layers[l]; i++) {
					int y = (i + 1) * h / (layers[l] + 1);
					pts[l][i] = new Point(x, y);
				}
			}
			return pts;
		}

		private void drawConnections(Graphics2D g2, int[] layers, Point[][] pts) {
			int wIdx = 0;

			for (int l = 0; l < layers.length - 1; l++) {
				for (int i = 0; i < layers[l]; i++) {
					for (int j = 0; j < layers[l + 1]; j++) {

						float w = nn.weights[wIdx++] * WEIGHT_SCALE;
						float a = Math.min(1f, Math.abs(w));

						g2.setStroke(new BasicStroke(1f + 4f * a));
						g2.setColor(weightColor(w, a));

						Point p1 = pts[l][i];
						Point p2 = pts[l + 1][j];
						g2.drawLine(p1.x, p1.y, p2.x, p2.y);
					}
				}
			}
		}

		private void drawNeurons(Graphics2D g2, int[] layers, Point[][] pts) {
			int bIdx = 0;
			int r = 8;

			for (int l = 1; l < layers.length; l++) {
				for (int i = 0; i < layers[l]; i++) {

					float b = nn.biases[bIdx++];
					float a = Math.min(1f, Math.abs(b));

					g2.setColor(biasColor(b, a));
					Point p = pts[l][i];
					g2.fillOval(p.x - r, p.y - r, r * 2, r * 2);
				}
			}
		}

		private Color weightColor(float v, float a) {
			return v >= 0 ? new Color(0f, 0f, 1f, a) : new Color(1f, 0f, 0f, a);
		}

		private Color biasColor(float v, float a) {
			return v >= 0 ? new Color(0f, 0.5f, 1f, a) : new Color(1f, 0.4f, 0.4f, a);
		}

	}

	public NNFrame() {
		super("...");
		super.setSize(400, 300);
		super.setVisible(true);
		super.setLayout(new BorderLayout());
		super.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public void addPanel(NNInstance inst) {
		super.getContentPane().removeAll();
		super.add(new NNPanel(inst), BorderLayout.CENTER);
		super.repaint();
	}

}
