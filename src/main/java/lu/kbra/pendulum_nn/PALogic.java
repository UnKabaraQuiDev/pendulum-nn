package lu.kbra.pendulum_nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.joml.Vector4f;

import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pendulum_nn.attrib_arrays.SyntheticFloatAttribArray;
import lu.kbra.pendulum_nn.attrib_arrays.SyntheticMat4fAttribArray;
import lu.kbra.pendulum_nn.attrib_arrays.SyntheticVec4fAttribArray;
import lu.kbra.pendulum_nn.shaders.compute.NNComputeComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.NNPostprocessComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearFloatComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearMat4fComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearVec4fComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.fill.FillVec4fComputeShader;
import lu.kbra.pendulum_nn.shaders.draw.DirectShader;
import lu.kbra.pendulum_nn.shaders.draw.InstanceDirectShader;
import lu.kbra.standalone.gameengine.GameEngine;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.geom.Mesh;
import lu.kbra.standalone.gameengine.geom.instance.InstanceEmitter;
import lu.kbra.standalone.gameengine.geom.utils.ObjLoader;
import lu.kbra.standalone.gameengine.graph.shader.RenderShader;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.scene.Scene3D;
import lu.kbra.standalone.gameengine.scene.camera.Camera;
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

	public static final boolean REAL_TIME = true;

	public static final float FIXED_D_TIME = 1f / 60;
	public static final int MAX_ITERATIONS = 60 * 60;
	public static final int TOP_AGENTS = 5;
	public static final int AGENT_BATCHES = 1;
	public static final int MAX_AGENTS = 10;

	public static final float MUTATE_RATE = 0.1f;
	public static final float MUTATE_STRENGTH = 0.1f;

	public static final float GRAVITY = 5f;
	public static final float MASS = 1f;
	public static final float HEAD_RADIUS = 0.14f;
	public static final float STARTING_ANGLE = (float) Math.PI / 1000 * 999;
	public static final float ACCELERATION_FACTOR = 2f;
	public static final float WIDTH = 3f;
	public static final boolean DEBUG_PERFECT_SCORE = false;

	public static int AGENT_PER_BATCHES;

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected FillVec4fComputeShader fillVec4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;
	protected NNPostprocessComputeShader nnPostprocessComputeShader;

	protected DirectShader directShader;
	protected InstanceDirectShader instanceDirectShader;
	protected Mesh mesh;
	protected LimitedInstanceEmitter instanceEmitter;
	protected Scene3D scene;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasesValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray transformsValueArray;
	protected SyntheticFloatAttribArray gradeNeuronsValueArray;

	protected final NNStructure struct = new NNStructure(5, new int[] { 1, 1, 1 }, 1, ActivationFunction.TANH);
	protected int instanceCount = 10;

//	protected NNFrame frame = new NNFrame();

	@Override
	public void init() throws Exception {
		// compute
		NNComputeComputeShader.LOCAL_SIZE = computeOptimalComputeShaderLocalSize();
		GlobalLogger.info("Compute local size: " + NNComputeComputeShader.LOCAL_SIZE);
		nnComputeComputeShader = new NNComputeComputeShader();
		AGENT_PER_BATCHES = NNComputeComputeShader.LOCAL_SIZE.x() * NNComputeComputeShader.LOCAL_SIZE.y()
				* NNComputeComputeShader.LOCAL_SIZE.z();

		assert AGENT_BATCHES * AGENT_PER_BATCHES * struct.computeWeightCount() < Integer.MAX_VALUE
				: AGENT_BATCHES * AGENT_PER_BATCHES * struct.computeWeightCount();
		assert AGENT_BATCHES * AGENT_PER_BATCHES * struct.computeBiasCount() < Integer.MAX_VALUE
				: AGENT_BATCHES * AGENT_PER_BATCHES * struct.computeBiasCount();

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		GlobalLogger.info("Compute local size: " + NNPostprocessComputeShader.LOCAL_SIZE);
		nnPostprocessComputeShader = new NNPostprocessComputeShader();

		instanceCount = Math.min(MAX_AGENTS, AGENT_BATCHES * AGENT_PER_BATCHES);
		GlobalLogger
				.info("Instancing: " + instanceCount + " instances on: " + AGENT_BATCHES + " * " + AGENT_PER_BATCHES + " for "
						+ NNComputeComputeShader.LOCAL_SIZE + " = " + nnComputeComputeShader.getGlobalGroup(instanceCount));

		final int weightCountTotal = struct.computeWeightCount() * instanceCount;
		weightsValueArray = new SyntheticFloatAttribArray("weights", WEIGHTS_IDX, weightCountTotal, BufferType.SHADER_STORAGE, false);

		final int biasCountTotal = struct.computeBiasCount() * instanceCount;
		biasesValueArray = new SyntheticFloatAttribArray("biases", BIASES_IDX, biasCountTotal, BufferType.SHADER_STORAGE, false);

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
		biasesValueArray.gen();
		biasesValueArray.init();

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

		fillVec4fComputeShader = new FillVec4fComputeShader();

		clear(weightsValueArray);
		clear(biasesValueArray);

		GlobalLogger.info("Cleared all buffers");

		final NNInstance inst = new NNInstance(struct, new float[struct.computeWeightCount()], new float[struct.computeBiasCount()]);
		final List<NNInstance> instances = new ArrayList<>();
		for (int i = 0; i < instanceCount; i++) {
			final NNInstance nInst = inst.clone();
			for (int j = 0; j < inst.getWeights().length; j++) {
				inst.getWeights()[j] = (float) (Math.random() * 2 - 1);
			}
			for (int j = 0; j < inst.getBiases().length; j++) {
				inst.getBiases()[j] = (float) (Math.random() * 2 - 1) * MUTATE_STRENGTH;
			}
			instances.add(nInst);
		}

		upload(instances);

		resetNNs();

		final float[] arr = physicsVec4sValueArray.read(0, physicsVec4sValueArray.getLength());
		System.err.println("Physics: " + Arrays.toString(arr));

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

		if (!REAL_TIME)
			startSelfRecurringTasks();
	}

	private void startSelfRecurringTasks() {
		for (int i = 0; i < 4; i++) {
			dispatchSelfRecurringTask();
		}
	}

	private void resetNNs() {
		fill(physicsVec4sValueArray, new Vector4f[] { new Vector4f(0, (float) Math.PI, 0, 0), new Vector4f(0, STARTING_ANGLE, 0, 0) });
		clear(transformsValueArray);
		clear(gradeNeuronsValueArray);
		clear(inputNeuronsValueArray);
		clear(outputNeuronsValueArray);

		// this sets up the inputs based on the physical state
		postProcess(FIXED_D_TIME);
	}

	private void upload(List<NNInstance> instances) {
		assert struct.computeNeuronCount() < NNComputeComputeShader.MAX_NEURONS;
		assert struct.getInnerLayers().length < NNComputeComputeShader.MAX_LAYERS;

		final float[] weights = PCUtils.pack(instances.parallelStream().map(NNInstance::getWeights).toArray(float[][]::new));
		assert weights.length == struct.computeWeightCount() * instanceCount : weights.length + " for " + instanceCount + " & "
				+ instances.size() + " = " + (struct.computeWeightCount() * instanceCount);
		weightsValueArray.update(weights);

		final float[] biases = PCUtils.pack(instances.parallelStream().map(NNInstance::getBiases).toArray(float[][]::new));
		assert biases.length == struct.computeBiasCount() * instanceCount
				: biases.length + " for " + instanceCount + " & " + instances.size() + " = " + (struct.computeBiasCount() * instanceCount);
		biasesValueArray.update(biases);
	}

	private void dispatchSelfRecurringTask() {
		RENDER_DISPATCHER.post(() -> {
			if (done.getValue()) {
				return;
			}
			compute();
			postProcess(FIXED_D_TIME);
			if (iterationCount.increment() < MAX_ITERATIONS) {
				iterationFrame.increment();
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
		RENDER_DISPATCHER.clear();

		{
			final float[] outs = outputNeuronsValueArray.read(0, outputNeuronsValueArray.getLength());
			System.err.println("Outputs: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays
					.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c)
					.summaryStatistics();
			System.err
					.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
							+ stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}
		{
			final float[] outs = physicsVec4sValueArray.read(0, physicsVec4sValueArray.getLength());
			System.err.println("Physics: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays
					.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c)
					.summaryStatistics();
			System.err
					.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
							+ stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}

		final float[] arr = gradeNeuronsValueArray.read(0, gradeNeuronsValueArray.getLength());
		System.err.println("Grades: " + Arrays.toString(arr));
		final DoubleSummaryStatistics stats = Arrays
				.stream(PCUtils.castObject(arr))
				.mapToDouble(c -> (double) (Float) (Object) c)
				.summaryStatistics();
		System.err
				.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
						+ stats.getMax() + " (stdDev) " + PCUtils.stdDev(arr));

		final int[] topIndices = PCUtils.getMaxIndices(arr, TOP_AGENTS);
		System.err.println("Keeping agents: " + Arrays.toString(topIndices));
		final List<NNInstance> topAgents = new ArrayList<>();
		for (int i = 0; i < topIndices.length; i++) {
			final float[] weights = weightsValueArray.read(i * struct.computeWeightCount(), struct.computeWeightCount());
			final float[] biases = biasesValueArray.read(i * struct.computeBiasCount(), struct.computeBiasCount());

			topAgents.add(new NNInstance(struct, weights, biases));
		}
//		frame.addPanel(topAgents.get(0));

		final List<NNInstance> allAgents = new ArrayList<>(topAgents);
		IntStream.range(0, instanceCount - topAgents.size()).forEach(i -> allAgents.add(mutate(topAgents.get(i % TOP_AGENTS).clone())));

		System.err.println("New: " + allAgents.size());
		upload(allAgents);

		resetNNs();

		done.setValue(false);
		iterationCount.set(0);
		if (!REAL_TIME)
			startSelfRecurringTasks();
	}

	Random rnd = new Random();

	private NNInstance mutate(NNInstance nn) {
		// Mutate weights
		for (int i = 0; i < nn.weights.length; i++) {
			if (rnd.nextFloat() < MUTATE_RATE) {
				nn.weights[i] = PCUtils.clamp(-10f, 10f, nn.weights[i] + (float) rnd.nextGaussian() * MUTATE_STRENGTH);
			}
		}

		// Mutate biases
		for (int i = 0; i < nn.biases.length; i++) {
			if (rnd.nextFloat() < MUTATE_RATE) {
				nn.biases[i] += rnd.nextGaussian() * MUTATE_STRENGTH;
			}
		}
		return nn;
	}

	private void compute() {
		nnComputeComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 1, weightsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 2, biasesValueArray.getGlId());
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
	protected IntPointer iterationFrame = new IntPointer(0);

	private void postProcess(float dTime) {
		nnPostprocessComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 4, physicsVec4sValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 5, transformsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 9, gradeNeuronsValueArray.getGlId());

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, dTime);
//		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INPUT_SIZE, struct.getInputCount());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, GRAVITY);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH, 1f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS, new Vector2f(-1, 1).mul(WIDTH / 2));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.FRICTION, 0.3f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ANGULAR_FRICTION, 0.2f);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_BOUNDS, new Vector2f(-1, 1));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_FACTOR, ACCELERATION_FACTOR);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_DIR, new Vector2f());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_SOURCE, new Vector2f());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.HEAD_RADIUS, HEAD_RADIUS);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.MASS, MASS);
		nnPostprocessComputeShader.setUniform("debugPerfectScore", DEBUG_PERFECT_SCORE);

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
		if (REAL_TIME) {
			compute();
			postProcess(dTime);
			final float[] output = outputNeuronsValueArray.read(0, outputNeuronsValueArray.getLength());
			System.err.println(Arrays.toString(output));
			final float[] grades = gradeNeuronsValueArray.read(0, gradeNeuronsValueArray.getLength());
			System.err.println(Arrays.toString(grades));
			if (iterationCount.increment() < MAX_ITERATIONS) {
				iterationFrame.increment();
			} else {
				readBack();
			}
		}

		System.err.println(iterationFrame.getValue() + " / " + iterationCount.getValue() + "/" + MAX_ITERATIONS);
		iterationFrame.set(0);

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
		if (biasesValueArray != null)
			biasesValueArray.cleanup();
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
		clearFloatComputeShader.setUniformUnsigned(ClearFloatComputeShader.ELEMENT_COUNT, array.getLength());
		final Vector3ic neededGlobalGroups = clearFloatComputeShader.getGlobalGroup(array.getLength());
//		System.err.println(neededGlobalGroups + " for: " + array.getLength());
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected void clear(SyntheticVec4fAttribArray array) {
		clearVec4fComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
		clearVec4fComputeShader.setUniformUnsigned(ClearVec4fComputeShader.ELEMENT_COUNT, array.getLength());
		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(array.getLength());
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected void clear(SyntheticMat4fAttribArray array) {
		clearMat4fComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
		clearMat4fComputeShader.setUniformUnsigned(ClearMat4fComputeShader.ELEMENT_COUNT, array.getLength());
		final Vector3ic neededGlobalGroups = clearMat4fComputeShader.getGlobalGroup(array.getLength());
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT);
	}

	protected void fill(SyntheticVec4fAttribArray array, Vector4f[] vals) {
		assert vals.length != 0;
		fillVec4fComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, array.getGlId());
//		AbstractShader.DEBUG = true;
		fillVec4fComputeShader.setUniform(FillVec4fComputeShader.DEFAULT_COUNT, vals.length);
		fillVec4fComputeShader.setUniform(FillVec4fComputeShader.DEFAULTS, vals);
		fillVec4fComputeShader.setUniformUnsigned(FillVec4fComputeShader.ELEMENT_COUNT, array.getLength());
		final Vector3ic neededGlobalGroups = fillVec4fComputeShader.getGlobalGroup(array.getLength());
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
