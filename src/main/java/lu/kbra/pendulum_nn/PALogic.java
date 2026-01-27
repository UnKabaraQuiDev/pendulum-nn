package lu.kbra.pendulum_nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.joml.Vector4f;

import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.standalone.gameengine.GameEngine;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.geom.Mesh;
import lu.kbra.standalone.gameengine.geom.instance.InstanceEmitter;
import lu.kbra.standalone.gameengine.geom.utils.ObjLoader;
import lu.kbra.standalone.gameengine.graph.shader.RenderShader;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.scene.Scene3D;
import lu.kbra.standalone.gameengine.scene.camera.Camera;
import lu.kbra.standalone.gameengine.utils.GameEngineUtils;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;
import lu.pcy113.pclib.PCUtils;
import lu.pcy113.pclib.logger.GlobalLogger;
import lu.pcy113.pclib.pointer.prim.BooleanPointer;
import lu.pcy113.pclib.pointer.prim.IntPointer;

public class PALogic extends GameLogic {

	public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = InstanceEmitter.TRANSFORM_BUFFER_INDEX;
	public static final int GRADE_IDX = InstanceEmitter.FIRST_BUFFER_INDEX;

	public static final float FIXED_D_TIME = 1f / 12;

	public static final int MAX_ITERATIONS = 10_000;

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;
	protected NNPostprocessComputeShader nnPostprocessComputeShader;

	protected DirectShader directShader;
	protected InstanceDirectShader instanceDirectShader;
	protected Mesh mesh;
	protected LimitedInstanceEmitter instanceEmitter;
	protected Scene3D scene;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray transformsValueArray;
	protected SyntheticFloatAttribArray gradeNeuronsValueArray;

	protected final NNStructure struct = new NNStructure(5, new int[] { 5, 4, 3, 2 }, 1, ActivationFunction.TANH);
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
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics", PHYSICS_IDX, physicsVec4sTotal, BufferType.SHADER_STORAGE, false,
				1);

		transformsValueArray = new SyntheticMat4fAttribArray("transforms", TRANSFORMS_IDX, instanceCount, BufferType.ARRAY, false);

		gradeNeuronsValueArray = new SyntheticFloatAttribArray("grade", GRADE_IDX, instanceCount, BufferType.SHADER_STORAGE, false);

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

		gradeNeuronsValueArray.gen();
		gradeNeuronsValueArray.init();

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
		clear(gradeNeuronsValueArray);

		GlobalLogger.info("Cleared all buffers");

		// compute

		NNComputeComputeShader.LOCAL_SIZE = computeOptimalComputeShaderLocalSize();
		GlobalLogger.info("Compute local size: " + NNComputeComputeShader.LOCAL_SIZE);
		nnComputeComputeShader = new NNComputeComputeShader();
		nnComputeComputeShader.bind();

		final NNInstance inst = new NNInstance(struct, new float[struct.computeWeightCount()], new float[struct.computeBiasCount()]);
		for (int i = 0; i < inst.getWeights().length; i++) {
			inst.getWeights()[i] = (float) (Math.random() * 2 - 1);
		}
		for (int i = 0; i < inst.getBiases().length; i++) {
			inst.getBiases()[i] = (float) (Math.random() * 2 - 1);
		}
		final List<NNInstance> instances = new ArrayList<>();

		final float[] weights = new float[struct.computeWeightCount() * instanceCount];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (float) (Math.random() * 2 - 1);
		}
//		System.arraycopy(inst.getWeights(), 0, weights, 0, inst.getWeights().length);
		weightsValueArray.update(weights);
		final float[] biases = new float[struct.computeBiasCount() * instanceCount];
		for (int i = 0; i < biases.length; i++) {
			biases[i] = (float) (Math.random() * 2 - 1);
		}
		biasValueArray.update(biases);

		final float[] inputs = new float[struct.getInputCount() * instanceCount];
		Arrays.fill(inputs, 0, struct.getInputCount(), 1);
		inputNeuronsValueArray.update(inputs);

		Vector4f[] state = new Vector4f[] { new Vector4f(0, 1, 0, 0), new Vector4f(0.1f, 0.2f, 0, 0) };
//		Arrays.fill(state, 0, 2, 1);
		state = GameEngineUtils.vec4Repeating(state, instanceCount);
		physicsVec4sValueArray.update(state);

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		GlobalLogger.info("Compute local size: " + NNPostprocessComputeShader.LOCAL_SIZE);
		nnPostprocessComputeShader = new NNPostprocessComputeShader();
		postProcess(FIXED_D_TIME);

		// draw
		directShader = new DirectShader();
		instanceDirectShader = new InstanceDirectShader();

		mesh = ObjLoader.loadMesh("pendulum", null, "classpath:/models/pendulum.obj");
		instanceEmitter = new LimitedInstanceEmitter("instances", mesh, instanceCount, transformsValueArray);

		scene = new Scene3D("scene");
		scene.setCamera(Camera.orthographicCamera3D());
		scene.getCamera().getProjection().setSize(0.4f);
		scene.getCamera().setPosition(new Vector3f(0, 10, 0));
		scene.getCamera().lookAt(scene.getCamera().getPosition(), GameEngine.ZERO, GameEngine.Z_POS);
		scene.getCamera().updateMatrix();

		for (int i = 0; i < 4; i++) {
			dispatchSelfRecurringTask();
		}
	}

	private void dispatchSelfRecurringTask() {
		RENDER_DISPATCHER.post(() -> {
			compute();
			postProcess(FIXED_D_TIME);
			if (iterationCount.increment() < MAX_ITERATIONS) {
				dispatchSelfRecurringTask();
			} else {
				readBack();
			}
		});
	}

	protected BooleanPointer done = new BooleanPointer(false);

	private void readBack() {
		if (done.getValue()) {
			return;
		}
		done.setValue(true);

		final float[] arr = new float[gradeNeuronsValueArray.getLength()];
		gradeNeuronsValueArray.bind();
		assert GL_W.glGetBufferParameteri(gradeNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length * Float.BYTES
				: arr.length + " & " + (arr.length * Float.BYTES) + " & "
						+ GL_W.glGetBufferParameteri(gradeNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE);
		GL_W.glGetBufferSubData(gradeNeuronsValueArray.getBufferType().getGlId(), 0, arr);
		GL_W.glFinish();
		System.err.println("Grades: " + Arrays.toString(arr));
	}

	private void compute() {
		nnComputeComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, weightsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, biasValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());

		nnComputeComputeShader.setUniform(NNComputeComputeShader.INPUT_SIZE, struct.getInputCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.LAYER_COUNT, struct.getInnerLayers().length + 1);
		nnComputeComputeShader
				.setUniform(NNComputeComputeShader.LAYER_SIZE,
						PCUtils.combineArrays(struct.getInnerLayers(), new int[] { struct.getOutputCount() }));
		nnComputeComputeShader.setUniform(NNComputeComputeShader.WEIGHT_OFFSET_PER_INSTANCE, struct.computeWeightCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.BIAS_OFFSET_PER_INSTANCE, struct.computeBiasCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.INSTANCE_COUNT, instanceCount);
		nnComputeComputeShader.setUniform(NNComputeComputeShader.ACTIVATION_FUNCTION, struct.getActivationFunction().ordinal());

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
//		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT | GL_W.GL_BUFFER_UPDATE_BARRIER_BIT);
//		GlobalLogger.info("Computed: " + instanceCount);

//		final float[] arr = new float[outputNeuronsValueArray.getLength()];
//		outputNeuronsValueArray.bind();
//		assert GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
//				* Float.BYTES
//				: arr.length + " & " + (arr.length * Float.BYTES) + " & "
//						+ GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE);
//		GL_W.glGetBufferSubData(outputNeuronsValueArray.getBufferType().getGlId(), 0, arr);
//		GL_W.glFinish();
//		System.err.println("Output neurons: " + Arrays.toString(arr));
	}

	protected IntPointer iterationCount = new IntPointer(0);

	private void postProcess(float dTime) {
		nnPostprocessComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 4, physicsVec4sValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 5, transformsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 9, gradeNeuronsValueArray.getGlId());

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, dTime);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INPUT_SIZE, struct.getInputCount());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, 9.81f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH, 1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS, new Vector2f(-1, 1));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.FRICTION, 0.98f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ANGULAR_FRICTION, 0.98f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_BOUNDS, new Vector2f(-1, 1));

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
//		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W
				.glMemoryBarrier(
						GL_W.GL_SHADER_STORAGE_BARRIER_BIT | GL_W.GL_BUFFER_UPDATE_BARRIER_BIT | GL_W.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
//		GlobalLogger.info("Post-processed: " + instanceCount);

//		final float[] arr = new float[physicsVec4sValueArray.getLength() * physicsVec4sValueArray.getElementComponentCount()];
//		physicsVec4sValueArray.bind();
//		assert GL_W.glGetBufferParameteri(physicsVec4sValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
//				* Float.BYTES;
//		GL_W.glGetBufferSubData(physicsVec4sValueArray.getBufferType().getGlId(), 0, arr);
//		GL_W.glFinish();
//		System.err.println("Physics: " + Arrays.toString(arr));
//
//		final float[] arr2 = new float[transformsValueArray.getLength() * transformsValueArray.getElementComponentCount()];
//		transformsValueArray.bind();
//		assert GL_W.glGetBufferParameteri(transformsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr2.length * Float.BYTES;
////		GL_W.glBindBuffer(GL_W.GL_SHADER_STORAGE_BUFFER, transformsValueArray.getGlId());
//		GL_W.glGetBufferSubData(transformsValueArray.getBufferType().getGlId(), 0, arr2);
//		GL_W.glFinish();
//		System.err.println("Transforms: " + Arrays.toString(arr2));
	}

	@Override
	public void input(float dTime) {

	}

	@Override
	public void update(float dTime) {

	}

	@Override
	public void render(float dTime) {
//		System.err.println(iterationCount.getValue());

		GL_W.glViewport(0, 0, window.getWidth(), window.getHeight());
		GL_W.glClearColor(0.1f, 0.2f, 0.3f, 1f);
		GL_W.glClear(GL_W.GL_DEPTH_BUFFER_BIT | GL_W.GL_COLOR_BUFFER_BIT);

		GL_W.glEnable(GL_W.GL_DEPTH_TEST);
		GL_W.glDisable(GL_W.GL_CULL_FACE);

		scene.getCamera().getProjection().update(window.getSize());

		instanceDirectShader.bind();
		mesh.bind();
		for (int i = 0; i < 4; i++) {
			GL_W.glVertexAttribDivisor(5 + i, 1);
			GL_W.glEnableVertexAttribArray(5 + i);
		}

		instanceDirectShader.setUniform(DirectShader.HAS_TEXTURE, false);
		instanceDirectShader.setUniformUnsigned(DirectShader.INSTANCE_COUNT, instanceEmitter.getParticleCount());
		instanceDirectShader.setUniform(RenderShader.TRANSFORMATION_MATRIX, GameEngine.IDENTITY_MATRIX4F);
		instanceDirectShader.setUniform(RenderShader.VIEW_MATRIX, scene.getCamera().getViewMatrix());
		instanceDirectShader.setUniform(RenderShader.PROJECTION_MATRIX, scene.getCamera().getProjection().getProjectionMatrix());

		GL_W
				.glDrawElementsInstanced(instanceDirectShader.getBeginMode().getGlId(),
						mesh.getIndicesCount(),
						GL_W.GL_UNSIGNED_INT,
						0,
						instanceEmitter.getParticleCount());
	}

	@Override
	public void cleanup() {
		if (mesh != null)
			mesh.cleanup();
		if (instanceEmitter != null)
			instanceEmitter.cleanup();
		if (clearFloatComputeShader != null)
			clearFloatComputeShader.cleanup();
		if (clearVec4fComputeShader != null)
			clearVec4fComputeShader.cleanup();
		if (clearMat4fComputeShader != null)
			clearMat4fComputeShader.cleanup();
		if (nnComputeComputeShader != null)
			nnComputeComputeShader.cleanup();
		if (nnPostprocessComputeShader != null)
			nnPostprocessComputeShader.cleanup();
		if (directShader != null)
			directShader.cleanup();
		if (instanceDirectShader != null)
			instanceDirectShader.cleanup();
		if (weightsValueArray != null)
			weightsValueArray.cleanup();
		if (biasValueArray != null)
			biasValueArray.cleanup();
		if (inputNeuronsValueArray != null)
			inputNeuronsValueArray.cleanup();
		if (outputNeuronsValueArray != null)
			outputNeuronsValueArray.cleanup();
		if (physicsVec4sValueArray != null)
			physicsVec4sValueArray.cleanup();
		if (transformsValueArray != null)
			transformsValueArray.cleanup();
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
