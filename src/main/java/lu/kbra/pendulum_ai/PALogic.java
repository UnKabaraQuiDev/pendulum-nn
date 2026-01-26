package lu.kbra.pendulum_ai;

import org.joml.Matrix4f;

import lu.kbra.standalone.gameengine.geom.Mesh;
import lu.kbra.standalone.gameengine.impl.GameLogic;

public class PALogic extends GameLogic {

	class NNStructure {
		int inputCount;
		int[] innerLayers;
		int outputCount;
	}

	class NNInstance {
		NNStructure struct;
		float[] weights;
		float[] biases;
		float[] input; // float[] output;
	}

	class NNMeshInstance {
		Mesh mesh;
		Matrix4f transform;
		float xPosition;
		float angleA;
		float velXPosition;
		float velAngleA;
	}

	@Override
	public void init() throws Exception {
		final NNStructure struct = null;
		int weightCount = 0;
		int prev = struct.inputCount;

		for (int l : struct.innerLayers) {
			weightCount += prev * l;
			prev = l;
		}
		weightCount += prev * struct.outputCount;
		final int weightSSBOBytes = weightCount * 4;

		int biasCount = 0;
		for (int l : struct.innerLayers)
			biasCount += l;
		biasCount += struct.outputCount;

		final int instanceCount = 10;

		int inputCountTotal = instanceCount * struct.inputCount;
		int inputSSBOBytes = inputCountTotal * 4;

		int outputCountTotal = instanceCount * struct.outputCount;
		int outputSSBOBytes = outputCountTotal * 4;

		int physicsVec4sPerInstance = 2;
		int physicsSSBOBytes = instanceCount * physicsVec4sPerInstance * 16;

		int transformSSBOBytes = instanceCount * 64;

	}

	@Override
	public void input(float dTime) {

	}

	@Override
	public void update(float dTime) {

	}

	@Override
	public void render(float dTime) {

	}

	@Override
	public void cleanup() {

	}

}
