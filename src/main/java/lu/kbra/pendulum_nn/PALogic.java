package lu.kbra.pendulum_nn;

import java.util.Arrays;

import org.joml.Vector2f;
import org.joml.Vector3i;
import org.joml.Vector3ic;

import com.fasterxml.jackson.databind.ObjectMapper;

import lu.pcy113.pclib.PCUtils;
import lu.pcy113.pclib.logger.GlobalLogger;

import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

public class PALogic extends GameLogic {

	public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = 5;

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;
	protected NNPostprocessComputeShader nnPostprocessComputeShader;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray transformsValueArray;

	protected final NNStructure struct = new NNStructure(5, new int[] { 5, 4, 3, 2 }, 1);
	protected final int instanceCount = 10;

	@Override
	public void init() throws Exception {
		final int weightCountTotal = struct.computeWeightCount() * instanceCount;
		weightsValueArray = new SyntheticFloatAttribArray("weights", WEIGHTS_IDX, weightCountTotal, BufferType.SHADER_STORAGE, false);

		final int biasCountTotal = struct.computeBiasCount() * instanceCount;
		biasValueArray = new SyntheticFloatAttribArray("biases", BIASES_IDX, biasCountTotal, BufferType.SHADER_STORAGE, false);

		final int inputCountTotal = instanceCount * struct.inputCount;
		inputNeuronsValueArray = new SyntheticFloatAttribArray("input", INPUT_IDX, inputCountTotal, BufferType.SHADER_STORAGE, false);

		final int outputCountTotal = instanceCount * struct.outputCount;
		outputNeuronsValueArray = new SyntheticFloatAttribArray("output", OUTPUT_IDX, outputCountTotal, BufferType.SHADER_STORAGE, false);

		final int physicsVec4sTotal = 2 * instanceCount;
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics", PHYSICS_IDX, physicsVec4sTotal, BufferType.SHADER_STORAGE, false);

		final int transformsTotal = instanceCount;
		transformsValueArray = new SyntheticMat4fAttribArray("transforms", TRANSFORMS_IDX, transformsTotal, BufferType.ARRAY, false);

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

		// compute

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

		compute();

		final float[] arr = new float[outputNeuronsValueArray.getLength()];
		assert GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
				* Float.BYTES;
		outputNeuronsValueArray.bind();
		GL_W.glGetBufferSubData(outputNeuronsValueArray.getBufferType().getGlId(), 0, arr);
		GL_W.glFinish();
		System.err.println(Arrays.toString(arr));

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		GlobalLogger.info("Compute local size: " + NNPostprocessComputeShader.LOCAL_SIZE);
		nnPostprocessComputeShader = new NNPostprocessComputeShader();

		postProcess();
	}

	private void compute() {
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
		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
		GlobalLogger.info("Computed: " + instanceCount);
	}

	private void postProcess() {
		nnPostprocessComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, outputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, physicsVec4sValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, transformsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, inputNeuronsValueArray.getGlId());

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, 0.1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INPUT_SIZE, struct.getInputCount());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, 9.81f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH, 1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS, new Vector2f(-1, 1));

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
		GlobalLogger.info("Post-processed: " + instanceCount);
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
