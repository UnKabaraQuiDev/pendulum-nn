package lu.kbra.pendulum_nn;

import java.util.Arrays;
import java.util.Objects;

public class NNStructure implements JacksonObject {

	protected int inputCount;
	protected int[] innerLayers;
	protected int outputCount;
	protected ActivationFunction activationFunction;

	public NNStructure() {
	}

	public NNStructure(int inputCount, int[] innerLayers, int outputCount, ActivationFunction activationFunction) {
		this.inputCount = inputCount;
		this.innerLayers = innerLayers;
		this.outputCount = outputCount;
		this.activationFunction = activationFunction;
	}

	public int computeWeightCount() {
		int total = 0;
		int prevSize = inputCount;

		for (int layerSize : innerLayers) {
			total += prevSize * layerSize;
			prevSize = layerSize;
		}
		total += prevSize * outputCount;
		return total;
	}

	public int computeBiasCount() {
		int total = 0;
		for (int layerSize : innerLayers) {
			total += layerSize;
		}
		total += outputCount;
		return total;
	}

	public int computeNeuronCount() {
		return inputCount + outputCount + Arrays.stream(innerLayers).sum();
	}

	public int getInputCount() {
		return inputCount;
	}

	public int[] getInnerLayers() {
		return innerLayers;
	}

	public int getOutputCount() {
		return outputCount;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public String toString() {
		return "NNStructure [inputCount=" + inputCount + ", innerLayers=" + Arrays.toString(innerLayers)
				+ ", outputCount=" + outputCount + ", activationFunction=" + activationFunction + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(innerLayers);
		result = prime * result + Objects.hash(activationFunction, inputCount, outputCount);
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
		NNStructure other = (NNStructure) obj;
		return activationFunction == other.activationFunction && Arrays.equals(innerLayers, other.innerLayers)
				&& inputCount == other.inputCount && outputCount == other.outputCount;
	}

}