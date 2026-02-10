package lu.kbra.pendulum_nn;

import java.util.Arrays;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonProperty;

public class NNInstance implements JacksonObject {

	protected float[] weights;
	protected float[] biases;

	public NNInstance() {
	}

	public NNInstance(float[] weights, float[] biases) {
		this.weights = weights;
		this.biases = biases;
	}

	public float[] getWeights() {
		return weights;
	}

	public float[] getBiases() {
		return biases;
	}

	@Override
	public String toString() {
		return "NNInstance@" + System.identityHashCode(this) + " [weights=" + Arrays.toString(weights) + ", biases="
				+ Arrays.toString(biases) + "]";
	}

	@Override
	public NNInstance clone() {
		return new NNInstance(weights.clone(), biases.clone());
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(biases);
		result = prime * result + Arrays.hashCode(weights);
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
		return Arrays.equals(biases, other.biases) && Arrays.equals(weights, other.weights);
	}

}