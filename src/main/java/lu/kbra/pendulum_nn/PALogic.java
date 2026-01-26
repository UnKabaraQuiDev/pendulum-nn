package lu.kbra.pendulum_nn;

import java.nio.FloatBuffer;
import java.util.Arrays;

import javax.swing.InputVerifier;

import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.lwjgl.system.MemoryUtil;

import lu.pcy113.pclib.PCUtils;
import lu.pcy113.pclib.logger.GlobalLogger;

import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

public class PALogic extends GameLogic {

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = 5;

	public static class NNStructure {

		protected int inputCount;
		protected int[] innerLayers;
		protected int outputCount;

		public NNStructure(int inputCount, int[] innerLayers, int outputCount) {
			this.inputCount = inputCount;
			this.innerLayers = innerLayers;
			this.outputCount = outputCount;
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

	}

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray transformsValueArray;

	@Override
	public void init() throws Exception {
		final int instanceCount = 10;

		final NNStructure struct = new NNStructure(3, new int[] { 2, 2 }, 1);
		int weightCount = 0;
		int prev = struct.inputCount;

		for (int l : struct.innerLayers) {
			weightCount += prev * l;
			prev = l;
		}
		weightCount += prev * struct.outputCount;
		final int totalWeightCount = weightCount * instanceCount;
		weightsValueArray = new SyntheticFloatAttribArray("weights", WEIGHTS_IDX, totalWeightCount, BufferType.SHADER_STORAGE, false);

		int biasCount = 0;
		for (int l : struct.innerLayers) {
			biasCount += l;
		}
		biasCount += struct.outputCount;
		final int totalBiasCount = biasCount * instanceCount;
		biasValueArray = new SyntheticFloatAttribArray("biases", BIASES_IDX, totalBiasCount, BufferType.SHADER_STORAGE, false);

		final int inputCountTotal = instanceCount * struct.inputCount;
		inputNeuronsValueArray = new SyntheticFloatAttribArray("input", INPUT_IDX, inputCountTotal, BufferType.SHADER_STORAGE, false);

		final int outputCountTotal = instanceCount * struct.outputCount;
		outputNeuronsValueArray = new SyntheticFloatAttribArray("output", OUTPUT_IDX, outputCountTotal, BufferType.SHADER_STORAGE, false);

		final int physicsVec4sPerInstance = 2;
		final int totalPhysicsVec4s = physicsVec4sPerInstance * instanceCount;
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics", PHYSICS_IDX, totalPhysicsVec4s, BufferType.SHADER_STORAGE, false);

		final int totalTransforms = instanceCount;
		transformsValueArray = new SyntheticMat4fAttribArray("transforms", TRANSFORMS_IDX, totalTransforms, BufferType.ARRAY, false);

		weightsValueArray.gen();
		weightsValueArray.init();
		biasValueArray.gen();
		biasValueArray.init();

		inputNeuronsValueArray.gen();
		inputNeuronsValueArray.init();
		outputNeuronsValueArray.gen();
		outputNeuronsValueArray.init();
		physicsVec4sValueArray.gen();
		physicsVec4sValueArray.init();

		transformsValueArray.gen();
		transformsValueArray.init();

		GlobalLogger.info("Created all buffers");

		clearFloatComputeShader = new ClearFloatComputeShader();
		clearVec4fComputeShader = new ClearVec4fComputeShader();
		clearMat4fComputeShader = new ClearMat4fComputeShader();

		clear(weightsValueArray);
		clear(biasValueArray);
		clear(inputNeuronsValueArray);
		clear(outputNeuronsValueArray);
		clear(physicsVec4sValueArray);
		clear(transformsValueArray);

		GlobalLogger.info("Cleared all buffers");

		NNComputeComputeShader.LOCAL_SIZE = computeOptimalComputeShaderLocalSize();
		GlobalLogger.info("Compute local size: " + NNComputeComputeShader.LOCAL_SIZE);
		nnComputeComputeShader = new NNComputeComputeShader();

		final float[] weights = new float[struct.computeWeightCount() * instanceCount];
		Arrays.fill(weights, 0, struct.computeWeightCount(), 1);
		weightsValueArray.update(weights);
		final float[] biases = new float[struct.computeBiasCount() * instanceCount];
		Arrays.fill(biases, 0, struct.computeWeightCount(), 0);
		biasValueArray.update(biases);

		final float[] inputs = new float[struct.getInputCount() * instanceCount];
		Arrays.fill(inputs, 0, struct.getInputCount(), 1);
		inputNeuronsValueArray.update(inputs);

		nnComputeComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, weightsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, biasValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());

		nnComputeComputeShader.setUniform(NNComputeComputeShader.INPUT_SIZE, struct.getInputCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.LAYER_COUNT, struct.getInnerLayers().length + 1);
		nnComputeComputeShader.setUniform(NNComputeComputeShader.LAYER_SIZE,
				PCUtils.combineArrays(struct.getInnerLayers(), new int[] { struct.getOutputCount() }));
		nnComputeComputeShader.setUniform(NNComputeComputeShader.WEIGHT_OFFSET_PER_INSTANCE, struct.computeWeightCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.BIAS_OFFSET_PER_INSTANCE, struct.computeBiasCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.INSTANCE_COUNT, instanceCount);

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);

		final float[] arr = new float[outputNeuronsValueArray.getLength()];
		assert GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
				* Float.BYTES;
		outputNeuronsValueArray.bind();
		GL_W.glGetBufferSubData(outputNeuronsValueArray.getBufferType().getGlId(), 0, arr);
		GL_W.glFlush();
		GL_W.glFinish();
		System.err.println(Arrays.toString(arr));
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

	protected void computeNN() {

	}

	protected void clear(SyntheticFloatAttribArray array) {
		clearFloatComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
		final Vector3ic neededGlobalGroups = clearFloatComputeShader.getGlobalGroup(new Vector3i(array.getLength()));
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected void clear(SyntheticVec4fAttribArray array) {
		clearVec4fComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(new Vector3i(array.getLength()));
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected void clear(SyntheticMat4fAttribArray array) {
		clearMat4fComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
		final Vector3ic neededGlobalGroups = clearMat4fComputeShader.getGlobalGroup(new Vector3i(array.getLength()));
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected Vector3ic computeOptimalComputeShaderLocalSize() {
		final int maxX[] = new int[1];
		GL_W.glGetIntegeri_v(GL_W.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, maxX);
		final int maxY[] = new int[1];
		GL_W.glGetIntegeri_v(GL_W.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, maxY);
		final int maxZ[] = new int[1];
		GL_W.glGetIntegeri_v(GL_W.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, maxZ);
		final int maxThreads = GL_W.glGetInteger(GL_W.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);

		final int localZ = Math.min(4, maxZ[0]);
		final int xyMax = maxThreads / localZ;
		final int localX = Math.min(16, maxX[0]);
		int localY = Math.min(xyMax / localX, maxY[0]);
		final int totalThreads = localX * localY * localZ;
		if (totalThreads > maxThreads) {
			localY = maxThreads / (localX * localZ);
		}
		return new Vector3i(localX, localY, localZ);
	}

}
