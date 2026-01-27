package lu.kbra.pendulum_nn;

import java.util.Arrays;

public class NNStructure {

	protected int inputCount;
	protected int[] innerLayers;
	protected int outputCount;
	protected ActivationFunction activationFunction;

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
		return "NNStructure [inputCount=" + inputCount + ", innerLayers=" + Arrays.toString(innerLayers) + ", outputCount=" + outputCount
				+ ", activationFunction=" + activationFunction + "]";
	}

}