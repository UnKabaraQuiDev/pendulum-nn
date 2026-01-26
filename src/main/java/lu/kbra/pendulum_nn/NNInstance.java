package lu.kbra.pendulum_nn;

import java.util.Arrays;

public class NNInstance {

	protected NNStructure structure;
	protected float[] weights;
	protected float[] biases;
	protected ActivationFunction activation;

	public NNInstance(NNStructure structure, float[] weights, float[] biases, ActivationFunction activation) {
		this.structure = structure;
		this.weights = weights;
		this.biases = biases;
		this.activation = activation;
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

	public ActivationFunction getActivation() {
		return activation;
	}

	@Override
	public String toString() {
		return "NNInstance@" + System.identityHashCode(this) + " [structure=" + structure + ", weights=" + Arrays.toString(weights)
				+ ", biases=" + Arrays.toString(biases) + ", activation=" + activation + "]";
	}

}