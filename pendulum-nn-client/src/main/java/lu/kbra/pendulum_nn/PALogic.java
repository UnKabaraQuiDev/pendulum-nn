package lu.kbra.pendulum_nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.joml.Math;
import org.joml.Vector2f;
import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.joml.Vector4f;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pclib.PCUtils;
import lu.kbra.pclib.datastructure.pair.Pairs;
import lu.kbra.pclib.datastructure.pair.ReadOnlyPair;
import lu.kbra.pclib.logger.GlobalLogger;
import lu.kbra.pclib.pointer.prim.BooleanPointer;
import lu.kbra.pclib.pointer.prim.IntPointer;
import lu.kbra.pendulum_nn.attrib_arrays.SyntheticFloatAttribArray;
import lu.kbra.pendulum_nn.attrib_arrays.SyntheticMat4fAttribArray;
import lu.kbra.pendulum_nn.attrib_arrays.SyntheticVec4fAttribArray;
import lu.kbra.pendulum_nn.shaders.compute.NNComputeComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.NNPostprocessComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearFloatComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearMat4fComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.clear.ClearVec4fComputeShader;
import lu.kbra.pendulum_nn.shaders.compute.fill.FillVec4fComputeShader;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.geom.instance.InstanceEmitter;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

@Component
public class PALogic extends GameLogic {

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = InstanceEmitter.TRANSFORM_BUFFER_INDEX;
	public static final int GRADE_IDX = InstanceEmitter.FIRST_BUFFER_INDEX;

	@Autowired
	private ObjectMapper objectMapper;
	@Autowired
	@Lazy
	private WSClient client;

	protected long simulationId;
	protected SimulationConfiguration simConfig;
	protected RunAgentsConfig runAgentsConfig;
	protected StartingCondition startingCondition;

	public static int AGENT_PER_BATCHES;

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected FillVec4fComputeShader fillVec4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;
	protected NNPostprocessComputeShader nnPostprocessComputeShader;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasesValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray baseTransformValueArray;
	protected SyntheticMat4fAttribArray armTransformValueArray;
	protected SyntheticMat4fAttribArray headTransformValueArray;
	protected SyntheticFloatAttribArray gradeNeuronsValueArray;

	protected NNStructure struct; // = new NNStructure(5, new int[] { 1, 1, 1 }, 1, ActivationFunction.TANH);
	protected int instanceCount = 10;

	final Vector2f prevA = new Vector2f();
	final Vector2f prevB = new Vector2f();

	@Override
	public void init() throws Exception {
//		MAIN_DISPATCHER.post(() -> GLFW.glfwHideWindow(((GLWindow) window).getHandle()));

		// compute
		NNComputeComputeShader.LOCAL_SIZE = computeOptimalComputeShaderLocalSize();
		GlobalLogger.info("Compute shader local size: " + NNComputeComputeShader.LOCAL_SIZE);
		nnComputeComputeShader = new NNComputeComputeShader();
		AGENT_PER_BATCHES = NNComputeComputeShader.LOCAL_SIZE.x() * NNComputeComputeShader.LOCAL_SIZE.y()
				* NNComputeComputeShader.LOCAL_SIZE.z();

//		assert simConfig.neuralNetworks.totalAgents * struct.computeWeightCount() < Integer.MAX_VALUE
//				: simConfig.neuralNetworks.totalAgents * struct.computeWeightCount();
//		assert simConfig.neuralNetworks.totalAgents * struct.computeBiasCount() < Integer.MAX_VALUE
//				: simConfig.neuralNetworks.totalAgents * struct.computeBiasCount();

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		nnPostprocessComputeShader = new NNPostprocessComputeShader();

//		instanceCount = simConfig.neuralNetworks.totalAgents;
//		GlobalLogger.info("Instancing: " + instanceCount + " instances on: "
//				+ (AGENT_PER_BATCHES / simConfig.neuralNetworks.totalAgents) + " * " + AGENT_PER_BATCHES + " for "
//				+ NNComputeComputeShader.LOCAL_SIZE + " = " + nnComputeComputeShader.getGlobalGroup(instanceCount));

		clearFloatComputeShader = new ClearFloatComputeShader();
		clearVec4fComputeShader = new ClearVec4fComputeShader();
		clearMat4fComputeShader = new ClearMat4fComputeShader();

		fillVec4fComputeShader = new FillVec4fComputeShader();

		GlobalLogger.info("Created shaders.");
	}

	private int totalGenerationCount = 1;
	private final IntPointer generationCount = new IntPointer(0);

	public void start() {
		while (RENDER_DISPATCHER == null) {
			Thread.onSpinWait();
		}

		RENDER_DISPATCHER.post(this::start_);
		generationCount.set(0);
	}

	protected void start_() {
		if (running.getValue()) {
			partialStop();
			GlobalLogger.info("Deleted previous buffers.");
		}

		this.instanceCount = runAgentsConfig.getTotalCount();
		this.totalGenerationCount = runAgentsConfig.totalGenerations;
		this.struct = runAgentsConfig.structure;

		final int weightCountTotal = struct.computeWeightCount() * instanceCount;
		weightsValueArray = new SyntheticFloatAttribArray("weights", WEIGHTS_IDX, weightCountTotal,
				BufferType.SHADER_STORAGE, false);

		final int biasCountTotal = struct.computeBiasCount() * instanceCount;
		biasesValueArray = new SyntheticFloatAttribArray("biases", BIASES_IDX, biasCountTotal,
				BufferType.SHADER_STORAGE, false);

		final int inputCountTotal = instanceCount * struct.inputCount;
		inputNeuronsValueArray = new SyntheticFloatAttribArray("input", INPUT_IDX, inputCountTotal,
				BufferType.SHADER_STORAGE, false);

		final int outputCountTotal = instanceCount * struct.outputCount;
		outputNeuronsValueArray = new SyntheticFloatAttribArray("output", OUTPUT_IDX, outputCountTotal,
				BufferType.SHADER_STORAGE, false);

		final int physicsVec4sTotal = 2 * instanceCount;
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics", PHYSICS_IDX, physicsVec4sTotal,
				BufferType.SHADER_STORAGE, false, 1);

		baseTransformValueArray = new SyntheticMat4fAttribArray("baseTransforms", TRANSFORMS_IDX, instanceCount,
				BufferType.ARRAY, false, 1);
		armTransformValueArray = new SyntheticMat4fAttribArray("armTransforms", TRANSFORMS_IDX, instanceCount,
				BufferType.ARRAY, false, 1);
		headTransformValueArray = new SyntheticMat4fAttribArray("headTransforms", TRANSFORMS_IDX, instanceCount,
				BufferType.ARRAY, false, 1);

		gradeNeuronsValueArray = new SyntheticFloatAttribArray("grade", GRADE_IDX, instanceCount,
				BufferType.SHADER_STORAGE, false);

		weightsValueArray.genInit();
		biasesValueArray.genInit();

		inputNeuronsValueArray.genInit();
		outputNeuronsValueArray.genInit();
		physicsVec4sValueArray.genInit();

		baseTransformValueArray.genInit();
		armTransformValueArray.genInit();
		headTransformValueArray.genInit();

		gradeNeuronsValueArray.genInit();

		GlobalLogger.info("Created all buffers.");

		clear(weightsValueArray);
		clear(biasesValueArray);
		clear(gradeNeuronsValueArray);

		GlobalLogger.info("Cleared all buffers.");

		if (runAgentsConfig.random) {
			final List<NNInstance> instances = new ArrayList<>(instanceCount);
			final NNInstance inst = new NNInstance(new float[struct.computeWeightCount()],
					new float[struct.computeBiasCount()]);
			for (int i = 0; i < instanceCount; i++) {
				final NNInstance nInst = inst.clone();
				for (int j = 0; j < inst.getWeights().length; j++) {
					nInst.getWeights()[j] = gaussianDelta(simConfig.neuralNetworks.mutateStrength);
				}
				for (int j = 0; j < inst.getBiases().length; j++) {
					nInst.getBiases()[j] = gaussianDelta(simConfig.neuralNetworks.mutateStrength);
				}
				instances.add(nInst);
			}
			upload(instances);
			GlobalLogger.info("Generated " + instanceCount + " random instances.");
		} else {
			upload(newAgents(runAgentsConfig.instances));
			GlobalLogger.info("Generated " + (instanceCount - runAgentsConfig.instances.size()) + " from "
					+ runAgentsConfig.instances.size() + " instances.");
		}

		resetNNs();

		running.set(true);
	}

	private void partialStop() {
		running.set(false);
		cleanup(weightsValueArray, biasesValueArray);
		cleanup(inputNeuronsValueArray, outputNeuronsValueArray, physicsVec4sValueArray);
		cleanup(baseTransformValueArray, armTransformValueArray, headTransformValueArray);
	}

	private float gaussianDelta(float scale) {
		return (float) (PCUtils.clamp(-1, 1, rand.nextGaussian()) * scale);
	}

	private void resetNNs() {
		startingCondition = new StartingCondition();

		if (simConfig.startState.fixedSimulationParameters) {
			fill(physicsVec4sValueArray,
					new Vector4f[] {
							new Vector4f(simConfig.startState.startingPosition, simConfig.startState.startingAngle, 0,
									0),
							new Vector4f(simConfig.startState.startingVelocity,
									simConfig.startState.startingAngleVelocity, 0, 0) });

			startingCondition.setxPosition(simConfig.startState.startingPosition);
			startingCondition.setAngle(simConfig.startState.startingAngle);
			startingCondition.setVelocity(simConfig.startState.startingVelocity);
			startingCondition.setAngularVelocity(simConfig.startState.startingAngleVelocity);
		} else {
			prevA.add(gaussianDelta(simConfig.startState.startingPositionScale),
					gaussianDelta(simConfig.startState.startingAngleScale));
			prevA.x = PCUtils.clamp(-simConfig.physics.width / 2, simConfig.physics.width / 2, prevA.x);
			prevB.add(gaussianDelta(simConfig.startState.startingVelocityScale),
					gaussianDelta(simConfig.startState.startingAngleVelocityScale));

			fill(physicsVec4sValueArray,
					new Vector4f[] { new Vector4f(prevA.x, prevA.y, 0, 0), new Vector4f(prevB.x, prevB.y, 0, 0) });

			startingCondition.setxPosition(prevA.x);
			startingCondition.setAngle(prevA.y);
			startingCondition.setVelocity(prevB.x);
			startingCondition.setAngularVelocity(prevB.y);
		}

//		clear(baseTransformValueArray);
//		clear(gradeNeuronsValueArray);
//		clear(inputNeuronsValueArray);
//		clear(outputNeuronsValueArray);

		// this sets up the inputs based on the physical state
		postProcess(simConfig.time.fixedDTime);
	}

	private void upload(List<NNInstance> instances) {
		assert struct.computeNeuronCount() < NNComputeComputeShader.MAX_NEURONS;
		assert struct.getInnerLayers().length < NNComputeComputeShader.MAX_LAYERS;

		final float[] weights = PCUtils
				.pack(instances.parallelStream().map(NNInstance::getWeights).toArray(float[][]::new));
		assert weights.length == struct.computeWeightCount() * instanceCount : weights.length + " for " + instanceCount
				+ " & " + instances.size() + " = " + (struct.computeWeightCount() * instanceCount);
		weightsValueArray.update(weights);

		final float[] biases = PCUtils
				.pack(instances.parallelStream().map(NNInstance::getBiases).toArray(float[][]::new));
		assert biases.length == struct.computeBiasCount() * instanceCount : biases.length + " for " + instanceCount
				+ " & " + instances.size() + " = " + (struct.computeBiasCount() * instanceCount);
		biasesValueArray.update(biases);
	}

	protected BooleanPointer done = new BooleanPointer(false);
	protected IntPointer totalGenerations = new IntPointer(0);

	protected Random rand = new Random();

	private void readBack() {
		if (done.getValue()) {
			return;
		}
		running.set(false);
		done.setValue(true);
		RENDER_DISPATCHER.clear();
		totalGenerations.increment();

		{
			final float[] outs = outputNeuronsValueArray.read(0, outputNeuronsValueArray.getLength());
//			System.err.println("Outputs: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c).summaryStatistics();
			System.err.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) "
					+ stats.getMin() + " (max) " + stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}
		{
			final float[] outs = physicsVec4sValueArray.read(0, physicsVec4sValueArray.getLength());
//			System.err.println("Physics: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c).summaryStatistics();
			System.err.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) "
					+ stats.getMin() + " (max) " + stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}

		final float[] arr = gradeNeuronsValueArray.read(0, gradeNeuronsValueArray.getLength());
//		System.err.println("Grades: " + Arrays.toString(arr));
		final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(arr))
				.mapToDouble(c -> (double) (Float) (Object) c).summaryStatistics();
		final double stdDev = PCUtils.stdDev(arr);
		final double median = PCUtils.median(arr);
		System.err.println("Stats: (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
				+ stats.getMax() + " (stdDev) " + stdDev + " (median) " + median);

		final int[] topIndices = PCUtils.getMaxIndices(arr, simConfig.neuralNetworks.topAgents);
		System.err.println("Keeping agents: "
				+ Arrays.stream(topIndices).mapToObj(Integer::toString).sorted().collect(Collectors.joining(", ")));
		final List<NNInstance> topAgents = new ArrayList<>();
		for (int index : topIndices) {
			final float[] weights = weightsValueArray.read(index * struct.computeWeightCount(),
					struct.computeWeightCount());
			final float[] biases = biasesValueArray.read(index * struct.computeBiasCount(), struct.computeBiasCount());
			topAgents.add(new NNInstance(weights, biases));
		}
		final List<NNInstance> allAgents = newAgents(topAgents);

		System.err.println("New: " + allAgents.size() + " for gen: " + totalGenerations.getValue());
		upload(allAgents);

		resetNNs();

		done.setValue(false);
		iterationCount.set(0);
		generationCount.decrement();
		sendData(IntStream.range(0, topIndices.length)
				.mapToObj(i -> Pairs.readOnly(arr[topIndices[i]], topAgents.get(i))).toList());

		if (generationCount.getValue() >= totalGenerationCount) {
			stop();
			System.err.println("end reached");
			return;
		}
		running.setValue(true);
	}

	private void sendData(List<ReadOnlyPair<Float, NNInstance>> list) {
		client.send(WSConsts.GEN_REPORT, new GenerationReport(simulationId, generationCount.getValue(), list,
				startingCondition, simConfig, struct));
		System.err.println("sent: " + new GenerationReport(simulationId, generationCount.getValue(), list,
				startingCondition, simConfig, struct));
	}

	private List<NNInstance> newAgents(List<NNInstance> topAgents) {
		final List<NNInstance> uniqueTopAgents = topAgents.stream().distinct().toList();
		if (uniqueTopAgents.size() < topAgents.size()) {
			System.err.println("Dropped " + (topAgents.size() - uniqueTopAgents.size()) + " duplicates");
		}
		System.err.println("Top agents hashes: " + uniqueTopAgents.parallelStream()
				.map(n -> Integer.toString(n.hashCode())).sorted().collect(Collectors.joining(", ")));

		final List<NNInstance> allAgents = new ArrayList<>(uniqueTopAgents);
		IntStream.range(0, instanceCount - uniqueTopAgents.size()).forEach(i -> {
			final NNInstance p1 = uniqueTopAgents.get(i % uniqueTopAgents.size());
			final NNInstance p2 = uniqueTopAgents
					.get((i + rand.nextInt(uniqueTopAgents.size())) % uniqueTopAgents.size());
			allAgents.add(mutate(crossover(p1, p2)));
		});

		return allAgents;
	}

	private NNInstance crossover(NNInstance p1, NNInstance p2) {
		final NNInstance child = p1.clone();

		for (int i = 0; i < child.weights.length; i++) {
			if (rand.nextBoolean()) {
				child.weights[i] = p2.weights[i];
			}
		}

		for (int i = 0; i < child.biases.length; i++) {
			if (rand.nextBoolean()) {
				child.biases[i] = p2.biases[i];
			}
		}

		return child;
	}

	private NNInstance mutate(NNInstance nn) {
		for (int i = 0; i < nn.weights.length; i++) {
			if (rand.nextFloat() < simConfig.neuralNetworks.mutateRate) {
				nn.weights[i] = PCUtils.clamp(-10f, 10f,
						nn.weights[i] + (float) rand.nextGaussian() / 40 * simConfig.neuralNetworks.mutateStrength);
			}
		}

		for (int i = 0; i < nn.biases.length; i++) {
			if (rand.nextFloat() < simConfig.neuralNetworks.mutateRate) {
				nn.biases[i] += rand.nextGaussian() * simConfig.neuralNetworks.mutateStrength;
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
		nnComputeComputeShader.setUniform(NNComputeComputeShader.LAYER_SIZE,
				PCUtils.combineArrays(struct.getInnerLayers(), new int[] { struct.getOutputCount() }));
		nnComputeComputeShader.setUniform(NNComputeComputeShader.WEIGHT_OFFSET_PER_INSTANCE,
				struct.computeWeightCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.BIAS_OFFSET_PER_INSTANCE, struct.computeBiasCount());
		nnComputeComputeShader.setUniform(NNComputeComputeShader.INSTANCE_COUNT, instanceCount);
		nnComputeComputeShader.setUniform(NNComputeComputeShader.ACTIVATION_FUNCTION,
				struct.getActivationFunction().ordinal());

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT | GL_W.GL_BUFFER_UPDATE_BARRIER_BIT);
	}

	protected IntPointer iterationCount = new IntPointer(0);
	protected BooleanPointer running = new BooleanPointer(false);

	private void postProcess(float dTime) {
		nnPostprocessComputeShader.bind();
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 3, outputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 4, physicsVec4sValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 5, baseTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 6, armTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 7, headTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 9, gradeNeuronsValueArray.getGlId());

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, dTime);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, simConfig.physics.gravity);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH,
				simConfig.physics.pendulumLength);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS,
				new Vector2f(-1, 1).mul(simConfig.physics.width / 2));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.FRICTION, simConfig.physics.friction);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ANGULAR_FRICTION,
				simConfig.physics.angularFriction);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_BOUNDS,
				new Vector2f(-simConfig.physics.accelerationMax, simConfig.physics.accelerationMax));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_FACTOR,
				simConfig.physics.accelerationFactor);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_DIR, new Vector2f());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_SOURCE, new Vector2f());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.MASS, simConfig.physics.mass);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.DEBUG_PERFECT_SCORE,
				simConfig.debugPerfectScore);

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(GL_W.GL_SHADER_STORAGE_BARRIER_BIT | GL_W.GL_BUFFER_UPDATE_BARRIER_BIT
				| GL_W.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	}

	@Override
	public void input(float dTime) {

	}

	@Override
	public void update(float dTime) {

	}

	@Override
	public void render(float dTime) {
		if (!running.getValue()) {
			Thread.onSpinWait();
			return;
		}
		for (int i = 0; i < simConfig.time.frameSubSteps; i++) {
			compute();
			postProcess(dTime);
			if (iterationCount.increment() < simConfig.time.maxIterations()) {
			} else {
				readBack();
				break;
			}
		}
	}

	@Override
	public void cleanup() {
		cleanup(weightsValueArray, biasesValueArray);
		cleanup(inputNeuronsValueArray, outputNeuronsValueArray, physicsVec4sValueArray);
		cleanup(baseTransformValueArray, armTransformValueArray, headTransformValueArray);

		cleanup(clearFloatComputeShader, clearVec4fComputeShader, clearMat4fComputeShader);
		cleanup(nnComputeComputeShader, nnPostprocessComputeShader);
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

	public void setSimConfig(SimulationConfiguration simConfig) {
		this.simConfig = simConfig;
	}

	public void setRunAgentsConfig(RunAgentsConfig runAgentsConfig) {
		this.runAgentsConfig = runAgentsConfig;
	}

	private void bigInfo(String string) {
		System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n");
		System.out.println("\t" + string + "\n\n\n");
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	}

}
