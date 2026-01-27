package lu.kbra.pendulum_nn;

import java.util.Arrays;

public class NNInstance {

	protected NNStructure structure;
	protected float[] weights;
	protected float[] biases;

	public NNInstance(NNStructure structure, float[] weights, float[] biases) {
		this.structure = structure;
		this.weights = weights;
		this.biases = biases;
	}

	public NNStructure getStructure() {
		return structure;
	}

	public float[] getWeights() {
		return weights;
	}

	public float[] getBiases() {
		return biases;
	}

	@Override
	public String toString() {
		return "NNInstance [structure=" + structure + ", weights=" + Arrays.toString(weights) + ", biases=" + Arrays.toString(biases) + "]";
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return new NNInstance(structure, weights, biases);
	}

}