package lu.kbra.pendulum_nn;

import java.util.Arrays;
import java.util.Objects;

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
	public NNInstance clone() {
		return new NNInstance(structure, weights.clone(), biases.clone());
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(biases);
		result = prime * result + Arrays.hashCode(weights);
		result = prime * result + Objects.hash(structure);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		NNInstance other = (NNInstance) obj;
		return Arrays.equals(biases, other.biases) && Objects.equals(structure, other.structure) && Arrays.equals(weights, other.weights);
	}

}