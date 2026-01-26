package lu.kbra.pendulum_nn;

import java.util.Arrays;

import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.joml.Vector4f;

import com.fasterxml.jackson.databind.ObjectMapper;

import lu.pcy113.pclib.PCUtils;
import lu.pcy113.pclib.logger.GlobalLogger;

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

public class PALogic extends GameLogic {

	public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = InstanceEmitter.TRANSFORM_BUFFER_INDEX;

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
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics", PHYSICS_IDX, physicsVec4sTotal, BufferType.SHADER_STORAGE, false, 1);

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
		nnComputeComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, weightsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, biasValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());

		final float[] weights = new float[struct.computeWeightCount() * instanceCount];
		Arrays.fill(weights, 0, struct.computeWeightCount(), 1);
		weightsValueArray.update(weights);
		final float[] biases = new float[struct.computeBiasCount() * instanceCount];
		Arrays.fill(biases, 0, struct.computeWeightCount(), 0);
		biasValueArray.update(biases);

		final float[] inputs = new float[struct.getInputCount() * instanceCount];
		Arrays.fill(inputs, 0, struct.getInputCount(), 1);
		inputNeuronsValueArray.update(inputs);

		Vector4f[] state = new Vector4f[] { new Vector4f(0, 1, 0, 0), new Vector4f(0.1f, 0.2f, 0, 0) };
//		Arrays.fill(state, 0, 2, 1);
		state = GameEngineUtils.vec4Repeating(state, instanceCount);
		physicsVec4sValueArray.update(state);

		compute();

//		final float[] arr = new float[outputNeuronsValueArray.getLength()];
//		assert GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
//				* Float.BYTES;
//		outputNeuronsValueArray.bind();
//		GL_W.glGetBufferSubData(outputNeuronsValueArray.getBufferType().getGlId(), 0, arr);
//		GL_W.glFinish();
//		System.err.println(Arrays.toString(arr));

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		GlobalLogger.info("Compute local size: " + NNPostprocessComputeShader.LOCAL_SIZE);
		nnPostprocessComputeShader = new NNPostprocessComputeShader();
		nnPostprocessComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, outputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, physicsVec4sValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, transformsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, inputNeuronsValueArray.getGlId());

		postProcess();

		// draw
		directShader = new DirectShader();
		instanceDirectShader = new InstanceDirectShader();

		mesh = ObjLoader.loadMesh("pendulum", null, "classpath:/models/pendulum.obj");
		instanceEmitter = new LimitedInstanceEmitter("instances", mesh, instanceCount, transformsValueArray);

		scene = new Scene3D("scene");
		scene.setCamera(Camera.perspectiveCamera3D());
		scene.getCamera().setPosition(new Vector3f(0, 10, 0));
		scene.getCamera().lookAt(scene.getCamera().getPosition(), GameEngine.ZERO, GameEngine.Z_POS);
		scene.getCamera().updateMatrix();
	}

	private void compute() {
		nnComputeComputeShader.bind();

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

		final float[] arr = new float[outputNeuronsValueArray.getLength()];
		assert GL_W.glGetBufferParameteri(outputNeuronsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
				* Float.BYTES;
		outputNeuronsValueArray.bind();
		GL_W.glGetBufferSubData(outputNeuronsValueArray.getBufferType().getGlId(), 0, arr);
		GL_W.glFinish();
		System.err.println("Output neurons: " + Arrays.toString(arr));
	}

	private void postProcess() {
		nnPostprocessComputeShader.bind();

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, 0.1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INPUT_SIZE, struct.getInputCount());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, 9.81f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH, 1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS, new Vector2f(-1, 1));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.FRICTION, 0.98f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ANGULAR_FRICTION, 0.98f);

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GlobalLogger.info("Computed: " + neededGlobalGroups + " for: " + instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
		GlobalLogger.info("Post-processed: " + instanceCount);

//		final float[] arr = new float[physicsVec4sValueArray.getLength() * physicsVec4sValueArray.getElementComponentCount()];
//		assert GL_W.glGetBufferParameteri(physicsVec4sValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr.length
//				* Float.BYTES;
//		physicsVec4sValueArray.bind();
//		GL_W.glGetBufferSubData(physicsVec4sValueArray.getBufferType().getGlId(), 0, arr);
//		GL_W.glFinish();
//		System.err.println("Physics: " + Arrays.toString(arr));

		final float[] arr2 = new float[transformsValueArray.getLength() * transformsValueArray.getElementComponentCount()];
		assert GL_W.glGetBufferParameteri(transformsValueArray.getBufferType().getGlId(), GL_W.GL_BUFFER_SIZE) == arr2.length * Float.BYTES;
		transformsValueArray.bind();
		GL_W.glGetBufferSubData(transformsValueArray.getBufferType().getGlId(), 0, arr2);
		GL_W.glFinish();
		System.err.println("Transforms: " + Arrays.toString(arr2));
	}

	@Override
	public void input(float dTime) {

	}

	@Override
	public void update(float dTime) {

	}

	@Override
	public void render(float dTime) {
		postProcess();

		GL_W.glViewport(0, 0, window.getWidth(), window.getHeight());
		GL_W.glClearColor(0.1f, 0.2f, 0.3f, 1f);
		GL_W.glClear(GL_W.GL_DEPTH_BUFFER_BIT | GL_W.GL_COLOR_BUFFER_BIT);

		GL_W.glEnable(GL_W.GL_DEPTH_TEST);
		GL_W.glDisable(GL_W.GL_CULL_FACE);

		scene.getCamera().getProjection().setPerspective(true);
//		scene.getCamera().getProjection().setSize(0.5f);
		scene.getCamera().getProjection().setFov((float) Math.toRadians(20f));
//		scene.getCamera().getPosition().x = 2;
//		scene.getCamera().getPosition().y = 20;
//		scene.getCamera().getPosition().z = 0;
		scene.getCamera().updateMatrix();
		scene.getCamera().getProjection().update(window.getSize());

		instanceDirectShader.bind();
		mesh.bind();

		instanceDirectShader.setUniform(DirectShader.HAS_TEXTURE, false);
//		instanceDirectShader.setUniform("tint", new Vector4f(1, 0.2f, 0.1f, 1));
		instanceDirectShader.setUniformUnsigned(DirectShader.INSTANCE_COUNT, instanceEmitter.getParticleCount());
		instanceDirectShader.setUniform(RenderShader.TRANSFORMATION_MATRIX, GameEngine.IDENTITY_MATRIX4F);
		instanceDirectShader.setUniform(RenderShader.VIEW_MATRIX, scene.getCamera().getViewMatrix());
		instanceDirectShader.setUniform(RenderShader.PROJECTION_MATRIX, scene.getCamera().getProjection().getProjectionMatrix());

//		System.err.println(instanceEmitter.getParticleCount());
		GL_W.glDrawElementsInstanced(instanceDirectShader.getBeginMode()
				.getGlId(), mesh.getIndicesCount(), GL_W.GL_UNSIGNED_INT, 0, instanceEmitter.getParticleCount());
	}

	@Override
	public void cleanup() {
		mesh.cleanup();
		instanceEmitter.cleanup();

		clearFloatComputeShader.cleanup();
		clearVec4fComputeShader.cleanup();
		clearMat4fComputeShader.cleanup();

		nnComputeComputeShader.cleanup();
		nnPostprocessComputeShader.cleanup();

		directShader.cleanup();
		instanceDirectShader.cleanup();

		weightsValueArray.cleanup();
		biasValueArray.cleanup();
		inputNeuronsValueArray.cleanup();
		outputNeuronsValueArray.cleanup();
		physicsVec4sValueArray.cleanup();
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
