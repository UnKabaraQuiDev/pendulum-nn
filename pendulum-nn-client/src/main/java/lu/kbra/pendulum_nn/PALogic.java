package lu.kbra.pendulum_nn;

import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.swing.SwingUtilities;

import org.joml.Math;
import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.joml.Vector3ic;
import org.joml.Vector4f;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pclib.PCUtils;
import lu.kbra.pclib.async.NextTask;
import lu.kbra.pclib.datastructure.pair.Pairs;
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
import lu.kbra.pendulum_nn.shaders.draw.DirectShader;
import lu.kbra.pendulum_nn.shaders.draw.InstanceDirectShader;
import lu.kbra.standalone.gameengine.GameEngine;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.geom.Mesh;
import lu.kbra.standalone.gameengine.geom.instance.InstanceEmitter;
import lu.kbra.standalone.gameengine.geom.utils.ObjLoader;
import lu.kbra.standalone.gameengine.graph.shader.RenderShader;
import lu.kbra.standalone.gameengine.impl.Cleanupable;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.kbra.standalone.gameengine.scene.Scene3D;
import lu.kbra.standalone.gameengine.scene.camera.Camera;
import lu.kbra.standalone.gameengine.utils.file.ShaderManager;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

public class PALogic extends GameLogic {

	public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
	public static final Path CONFIG_DIR = Paths.get("./config/");
	public static final File CONFIG_FILE = CONFIG_DIR.resolve("sim.json").toFile();

	public static final int INPUT_IDX = 0;
	public static final int OUTPUT_IDX = 1;
	public static final int PHYSICS_IDX = 2;
	public static final int WEIGHTS_IDX = 3;
	public static final int BIASES_IDX = 4;
	public static final int TRANSFORMS_IDX = InstanceEmitter.TRANSFORM_BUFFER_INDEX;
	public static final int GRADE_IDX = InstanceEmitter.FIRST_BUFFER_INDEX;

	public SimulationConfiguration simConfig;

	public static int AGENT_PER_BATCHES;

	protected ClearFloatComputeShader clearFloatComputeShader;
	protected ClearVec4fComputeShader clearVec4fComputeShader;
	protected ClearMat4fComputeShader clearMat4fComputeShader;

	protected FillVec4fComputeShader fillVec4fComputeShader;

	protected NNComputeComputeShader nnComputeComputeShader;
	protected NNPostprocessComputeShader nnPostprocessComputeShader;

	protected DirectShader directShader;
	protected InstanceDirectShader instanceDirectShader;
	protected Optional<ShaderManager> shaderManager;

	protected Scene3D scene;

	protected Mesh pendulumBaseMesh;
	protected LimitedInstanceEmitter baseInstances;
	protected Mesh pendulumArmMesh;
	protected LimitedInstanceEmitter armInstances;
	protected Mesh pendulumHeadMesh;
	protected LimitedInstanceEmitter headInstances;
	protected Mesh triangleMesh;
	protected LimitedInstanceEmitter triangleEmitter;

	protected SyntheticFloatAttribArray weightsValueArray;
	protected SyntheticFloatAttribArray biasesValueArray;
	protected SyntheticFloatAttribArray inputNeuronsValueArray;
	protected SyntheticFloatAttribArray outputNeuronsValueArray;
	protected SyntheticVec4fAttribArray physicsVec4sValueArray;
	protected SyntheticMat4fAttribArray baseTransformValueArray;
	protected SyntheticMat4fAttribArray armTransformValueArray;
	protected SyntheticMat4fAttribArray headTransformValueArray;
	protected SyntheticFloatAttribArray gradeNeuronsValueArray;

	protected final NNStructure struct = new NNStructure(5, new int[] { 1, 1, 1 }, 1, ActivationFunction.TANH);
	protected int instanceCount = 10;

	protected NNFrame frame = new NNFrame();

	final Vector2f prevA = new Vector2f();
	final Vector2f prevB = new Vector2f();

	@Deprecated
	public static List<NNInstance> loadLatestOutput() throws IOException {
		final File baseDir = new File("./output");
		if (!baseDir.isDirectory()) {
			throw new IllegalStateException("output directory not found");
		}

		final Optional<File> largestDir = Files.list(baseDir.toPath())
				.map(java.nio.file.Path::toFile)
				.filter(File::isDirectory)
				.filter(f -> f.getName().matches("\\d+"))
				.max(Comparator.comparingLong(f -> Long.parseLong(f.getName())));

		if (largestDir.isEmpty()) {
			throw new IllegalStateException("no numeric directories found");
		}

		final Pattern pattern = Pattern.compile("top\\.(\\d+)\\.json");

		final Optional<File> largestFile = Files.list(largestDir.get().toPath())
				.map(java.nio.file.Path::toFile)
				.filter(File::isFile)
				.filter(f -> pattern.matcher(f.getName()).matches())
				.max(Comparator.comparingLong(f -> {
					Matcher m = pattern.matcher(f.getName());
					m.find();
					return Long.parseLong(m.group(1));
				}));

		if (largestFile.isEmpty()) {
			throw new IllegalStateException("no matching json files found");
		}

		System.err.println("Loading from: " + largestFile.get().getPath());

		return OBJECT_MAPPER.readValue(largestFile.get(), new TypeReference<List<NNInstance>>() {
		});
	}

	@Override
	public void init() throws Exception {
		if (!CONFIG_FILE.exists()) {
			CONFIG_FILE.getParentFile().mkdirs();
			OBJECT_MAPPER.writeValue(CONFIG_FILE, new SimulationConfiguration(true));
			bigInfo("The config file was created to: " + CONFIG_FILE.getAbsolutePath()
					+ ", review the config and restart the program to start.");
			super.stop();
		}
		simConfig = OBJECT_MAPPER.readValue(CONFIG_FILE, SimulationConfiguration.class);

		if (simConfig.hotReloadShaders) {
			shaderManager = Optional.of(new ShaderManager(cache, "src/main/resources/"));
		} else {
			shaderManager = Optional.empty();
		}

		// compute
		NNComputeComputeShader.LOCAL_SIZE = computeOptimalComputeShaderLocalSize();
		GlobalLogger.info("Compute shader local size: " + NNComputeComputeShader.LOCAL_SIZE);
		nnComputeComputeShader = new NNComputeComputeShader();
		AGENT_PER_BATCHES = NNComputeComputeShader.LOCAL_SIZE.x() * NNComputeComputeShader.LOCAL_SIZE.y()
				* NNComputeComputeShader.LOCAL_SIZE.z();

		assert simConfig.neuralNetworks.totalAgents * struct.computeWeightCount() < Integer.MAX_VALUE
				: simConfig.neuralNetworks.totalAgents * struct.computeWeightCount();
		assert simConfig.neuralNetworks.totalAgents * struct.computeBiasCount() < Integer.MAX_VALUE
				: simConfig.neuralNetworks.totalAgents * struct.computeBiasCount();

		// post process
		NNPostprocessComputeShader.LOCAL_SIZE = NNComputeComputeShader.LOCAL_SIZE;
		GlobalLogger.info("Compute local size: " + NNPostprocessComputeShader.LOCAL_SIZE);
		nnPostprocessComputeShader = new NNPostprocessComputeShader();
		shaderManager.ifPresent(c -> c.monitorShader(nnPostprocessComputeShader));

		instanceCount = simConfig.neuralNetworks.totalAgents;
		GlobalLogger.info("Instancing: " + instanceCount + " instances on: " + (AGENT_PER_BATCHES / simConfig.neuralNetworks.totalAgents)
				+ " * " + AGENT_PER_BATCHES + " for " + NNComputeComputeShader.LOCAL_SIZE + " = "
				+ nnComputeComputeShader.getGlobalGroup(instanceCount));

		final int weightCountTotal = struct.computeWeightCount() * instanceCount;
		weightsValueArray = new SyntheticFloatAttribArray("weights", WEIGHTS_IDX, weightCountTotal, BufferType.SHADER_STORAGE, false);

		final int biasCountTotal = struct.computeBiasCount() * instanceCount;
		biasesValueArray = new SyntheticFloatAttribArray("biases", BIASES_IDX, biasCountTotal, BufferType.SHADER_STORAGE, false);

		final int inputCountTotal = instanceCount * struct.inputCount;
		inputNeuronsValueArray = new SyntheticFloatAttribArray("input", INPUT_IDX, inputCountTotal, BufferType.SHADER_STORAGE, false);

		final int outputCountTotal = instanceCount * struct.outputCount;
		outputNeuronsValueArray = new SyntheticFloatAttribArray("output", OUTPUT_IDX, outputCountTotal, BufferType.SHADER_STORAGE, false);

		final int physicsVec4sTotal = 2 * instanceCount;
		physicsVec4sValueArray = new SyntheticVec4fAttribArray("physics",
				PHYSICS_IDX,
				physicsVec4sTotal,
				BufferType.SHADER_STORAGE,
				false,
				1);

		baseTransformValueArray = new SyntheticMat4fAttribArray("baseTransforms",
				TRANSFORMS_IDX,
				instanceCount,
				BufferType.ARRAY,
				false,
				1);
		armTransformValueArray = new SyntheticMat4fAttribArray("armTransforms", TRANSFORMS_IDX, instanceCount, BufferType.ARRAY, false, 1);
		headTransformValueArray = new SyntheticMat4fAttribArray("headTransforms",
				TRANSFORMS_IDX,
				instanceCount,
				BufferType.ARRAY,
				false,
				1);

		gradeNeuronsValueArray = new SyntheticFloatAttribArray("grade", GRADE_IDX, instanceCount, BufferType.SHADER_STORAGE, false);

		weightsValueArray.genInit();
		biasesValueArray.genInit();

		inputNeuronsValueArray.genInit();
		outputNeuronsValueArray.genInit();
		physicsVec4sValueArray.genInit();

		baseTransformValueArray.genInit();
		armTransformValueArray.genInit();
		headTransformValueArray.genInit();

		gradeNeuronsValueArray.genInit();

		GlobalLogger.info("Created all buffers");

		clearFloatComputeShader = new ClearFloatComputeShader();
		clearVec4fComputeShader = new ClearVec4fComputeShader();
		clearMat4fComputeShader = new ClearMat4fComputeShader();

		fillVec4fComputeShader = new FillVec4fComputeShader();

		clear(weightsValueArray);
		clear(biasesValueArray);
		clear(gradeNeuronsValueArray);

		GlobalLogger.info("Cleared all buffers");

		final List<NNInstance> instances = new ArrayList<>();
		if (simConfig.persistance.reloadSpecific) {
			assert false;
		} else if (simConfig.persistance.reloadLatest) {
			instances.addAll(loadLatestOutput());
			assert instances.get(0).getStructure().equals(struct) : struct + " <> " + instances.get(0).getStructure();
			System.err.println("Reloaded: " + instances);
			IntStream.range(0, instanceCount - instances.size()).forEach(i -> {
				final NNInstance p1 = instances.get(i % instances.size());
				final NNInstance p2 = instances.get((i + rand.nextInt(instances.size())) % instances.size());
				instances.add(mutate(crossover(p1, p2)));
			});
		} else {
			final NNInstance inst = new NNInstance(struct, new float[struct.computeWeightCount()], new float[struct.computeBiasCount()]);
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
		}

		upload(instances);

		prevA.set(simConfig.startState.startingPosition, simConfig.startState.startingAngle);
		prevB.set(simConfig.startState.startingVelocity, simConfig.startState.startingAngleVelocity);

		resetNNs();

		// draw
		directShader = new DirectShader();
		instanceDirectShader = new InstanceDirectShader();

		pendulumBaseMesh = ObjLoader.loadMesh("pendulum_base", null, "classpath:/models/pendulum_base.obj");
		baseInstances = new LimitedInstanceEmitter("baseInstances", pendulumBaseMesh, instanceCount, baseTransformValueArray);
		pendulumArmMesh = ObjLoader.loadMesh("pendulum_arm", null, "classpath:/models/pendulum_arm.obj");
		armInstances = new LimitedInstanceEmitter("armInstances", pendulumArmMesh, instanceCount, armTransformValueArray);
		pendulumHeadMesh = ObjLoader.loadMesh("pendulum_head", null, "classpath:/models/pendulum_head.obj");
		headInstances = new LimitedInstanceEmitter("headInstances", pendulumHeadMesh, instanceCount, headTransformValueArray);

		triangleMesh = ObjLoader.loadMesh("triangle", null, "classpath:/models/triangle.obj");
		triangleEmitter = new LimitedInstanceEmitter("triangles", triangleMesh, instanceCount, headTransformValueArray);

		scene = new Scene3D("scene");
		scene.setCamera(Camera.orthographicCamera3D());
		scene.getCamera().getProjection().setSize(0.4f);
		scene.getCamera().setPosition(new Vector3f(0, -10, 0));
		scene.getCamera().lookAt(scene.getCamera().getPosition(), GameEngine.ZERO, GameEngine.Z_POS);
		scene.getCamera().updateMatrix();

		if (!simConfig.time.realTime) {
			startSelfRecurringTasks();
		}
	}

	private void bigInfo(String string) {
		System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n");
		System.out.println("\t" + string + "\n\n\n");
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	}

	private void startSelfRecurringTasks() {
		for (int i = 0; i < 4; i++) {
			dispatchSelfRecurringTask();
		}
	}

	private float gaussianDelta(float scale) {
		return (float) (PCUtils.clamp(-1, 1, rand.nextGaussian()) * scale);
	}

	private void resetNNs() {
		if (simConfig.startState.fixedSimulationParameters) {
			fill(physicsVec4sValueArray,
					new Vector4f[] {
							new Vector4f(simConfig.startState.startingPosition, simConfig.startState.startingAngle, 0, 0),
							new Vector4f(simConfig.startState.startingVelocity, simConfig.startState.startingAngleVelocity, 0, 0) });

			frame.getStartingPosHistory()
					.add(PCUtils.map(simConfig.startState.startingPosition,
							-simConfig.physics.width / 2,
							simConfig.physics.width / 2,
							0.0,
							100));
			frame.getStartingVelHistory()
					.add(PCUtils.map(simConfig.startState.startingVelocity,
							-simConfig.physics.accelerationMax,
							simConfig.physics.accelerationMax,
							0.0,
							100));
			frame.getStartingAngleHistory().add(PCUtils.map(simConfig.startState.startingAngle, -Math.PI, Math.PI, 0.0, 100));
			frame.getStartingAngleVelHistory()
					.add(PCUtils.map(simConfig.startState.startingAngleVelocity,
							-simConfig.physics.accelerationMax,
							simConfig.physics.accelerationMax,
							0.0,
							100));
		} else {
			prevA.add(gaussianDelta(simConfig.startState.startingPositionScale), gaussianDelta(simConfig.startState.startingAngleScale));
			prevA.x = PCUtils.clamp(-simConfig.physics.width / 2, simConfig.physics.width / 2, prevA.x);
			prevB.add(gaussianDelta(simConfig.startState.startingVelocityScale),
					gaussianDelta(simConfig.startState.startingAngleVelocityScale));

			System.err.println("Starting conditions: " + prevA.x + ", " + prevA.y + " & " + prevB.x + ", " + prevB.y);

			fill(physicsVec4sValueArray, new Vector4f[] { new Vector4f(prevA.x, prevA.y, 0, 0), new Vector4f(prevB.x, prevB.y, 0, 0) });

			frame.getStartingPosHistory().add(PCUtils.map(prevA.x, -simConfig.physics.width / 2, simConfig.physics.width / 2, 0.0, 100));
			frame.getStartingVelHistory()
					.add(PCUtils.map(prevA.y, -simConfig.physics.accelerationMax, simConfig.physics.accelerationMax, 0.0, 100));
			frame.getStartingAngleHistory().add(PCUtils.map(prevB.x, -Math.PI, Math.PI, 0.0, 100));
			frame.getStartingAngleVelHistory()
					.add(PCUtils.map(prevB.y, -simConfig.physics.accelerationMax, simConfig.physics.accelerationMax, 0.0, 100));
		}
		SwingUtilities.invokeLater(frame::repaint);

		clear(baseTransformValueArray);
		clear(gradeNeuronsValueArray);
		clear(inputNeuronsValueArray);
		clear(outputNeuronsValueArray);

		// this sets up the inputs based on the physical state
		postProcess(simConfig.time.fixedDTime);
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
			postProcess(simConfig.time.fixedDTime);
			if (iterationCount.increment() < simConfig.time.maxIterations()) {
				iterationFrame.increment();
				dispatchSelfRecurringTask();
			} else {
				readBack();
			}
		});
	}

	protected BooleanPointer done = new BooleanPointer(false);
	protected IntPointer totalGenerations = new IntPointer(0);

	protected Random rand = new Random();

	private void readBack() {
		if (done.getValue()) {
			return;
		}
		done.setValue(true);
		RENDER_DISPATCHER.clear();
		totalGenerations.increment();

		{
			final float[] outs = outputNeuronsValueArray.read(0, outputNeuronsValueArray.getLength());
//			System.err.println("Outputs: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c)
					.summaryStatistics();
			System.err.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
					+ stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}
		{
			final float[] outs = physicsVec4sValueArray.read(0, physicsVec4sValueArray.getLength());
//			System.err.println("Physics: " + Arrays.toString(outs));
			final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(outs))
					.mapToDouble(c -> (double) (Float) (Object) c)
					.summaryStatistics();
			System.err.println("Stats: (sum) " + stats.getSum() + " (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) "
					+ stats.getMax() + " (stdDev) " + PCUtils.stdDev(outs));
		}

		final float[] arr = gradeNeuronsValueArray.read(0, gradeNeuronsValueArray.getLength());
//		System.err.println("Grades: " + Arrays.toString(arr));
		final DoubleSummaryStatistics stats = Arrays.stream(PCUtils.castObject(arr))
				.mapToDouble(c -> (double) (Float) (Object) c)
				.summaryStatistics();
		final double stdDev = PCUtils.stdDev(arr);
		final double median = PCUtils.median(arr);
		System.err.println("Stats: (avg) " + stats.getAverage() + " (min) " + stats.getMin() + " (max) " + stats.getMax() + " (stdDev) "
				+ stdDev + " (median) " + median);

		frame.getAvgHistory().add(stats.getAverage());
		frame.getMaxHistory().add(stats.getMax());
		frame.getMinHistory().add(stats.getMin());
		frame.getStdDevHistory().add(Pairs.readOnly(stats.getAverage() - stdDev, stats.getAverage() + stdDev));
		SwingUtilities.invokeLater(frame::repaint);

		final int[] topIndices = PCUtils.getMaxIndices(arr, simConfig.neuralNetworks.topAgents);
		System.err.println(
				"Keeping agents: " + Arrays.stream(topIndices).mapToObj(Integer::toString).sorted().collect(Collectors.joining(", ")));
		final List<NNInstance> topAgents = new ArrayList<>();
		for (int index : topIndices) {
			final float[] weights = weightsValueArray.read(index * struct.computeWeightCount(), struct.computeWeightCount());
			final float[] biases = biasesValueArray.read(index * struct.computeBiasCount(), struct.computeBiasCount());
			topAgents.add(new NNInstance(struct, weights, biases));
		}
		final List<NNInstance> uniqueTopAgents = topAgents.stream().distinct().toList();
		if (uniqueTopAgents.size() < topAgents.size()) {
			System.err.println("Dropped " + (topAgents.size() - uniqueTopAgents.size()) + " duplicates");
		}
		System.err.println("Top agents hashes: "
				+ uniqueTopAgents.parallelStream().map(n -> Integer.toString(n.hashCode())).sorted().collect(Collectors.joining(", ")));

		NextTask.create(() -> {
			RuntimeMXBean bean = ManagementFactory.getRuntimeMXBean();
			final File f = new File("./output/" + bean.getStartTime() + "/top."
					+ PCUtils.leftPadString(Integer.toString(totalGenerations.getValue()), "0", 5) + ".json");
			f.getParentFile().mkdirs();
			OBJECT_MAPPER.writeValue(f, uniqueTopAgents);
		}).catch_(Throwable::printStackTrace).runAsync();

		triangleEmitter.setParticleCount(uniqueTopAgents.size());

		final List<NNInstance> allAgents = new ArrayList<>(uniqueTopAgents);
		IntStream.range(0, instanceCount - uniqueTopAgents.size()).forEach(i -> {
			final NNInstance p1 = uniqueTopAgents.get(i % uniqueTopAgents.size());
			final NNInstance p2 = uniqueTopAgents.get((i + rand.nextInt(uniqueTopAgents.size())) % uniqueTopAgents.size());
			allAgents.add(mutate(crossover(p1, p2)));
		});

		System.err.println("New: " + allAgents.size() + " for gen: " + totalGenerations.getValue());
		upload(allAgents);

		resetNNs();

		done.setValue(false);
		iterationCount.set(0);
		if (!simConfig.time.realTime)
			startSelfRecurringTasks();
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
				nn.weights[i] = PCUtils
						.clamp(-10f, 10f, nn.weights[i] + (float) rand.nextGaussian() / 40 * simConfig.neuralNetworks.mutateStrength);
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
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 5, baseTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 6, armTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 7, headTransformValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 0, inputNeuronsValueArray.getGlId());
		GL_W.glBindBufferBase(BufferType.SHADER_STORAGE.getGlId(), 9, gradeNeuronsValueArray.getGlId());

		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.D_TIME, dTime);
//		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INPUT_SIZE, struct.getInputCount());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.INSTANCE_COUNT, instanceCount);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.GRAVITY, simConfig.physics.gravity);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.PENDULUM_LENGTH, simConfig.physics.pendulumLength);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.BOUNDS, new Vector2f(-1, 1).mul(simConfig.physics.width / 2));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.FRICTION, simConfig.physics.friction);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ANGULAR_FRICTION, simConfig.physics.angularFriction);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_BOUNDS,
				new Vector2f(-simConfig.physics.accelerationMax, simConfig.physics.accelerationMax));
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.ACCELERATION_FACTOR, simConfig.physics.accelerationFactor);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_DIR, new Vector2f());
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.USER_FORCE_SOURCE, new Vector2f());
//		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.HEAD_RADIUS, simConfig.physics.rad);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.MASS, simConfig.physics.mass);
		nnPostprocessComputeShader.setUniform(NNPostprocessComputeShader.DEBUG_PERFECT_SCORE, simConfig.debugPerfectScore);

		final Vector3ic neededGlobalGroups = clearVec4fComputeShader.getGlobalGroup(instanceCount);
		GL_W.glDispatchCompute(neededGlobalGroups.x(), neededGlobalGroups.y(), neededGlobalGroups.z());
		GL_W.glMemoryBarrier(
				GL_W.GL_SHADER_STORAGE_BARRIER_BIT | GL_W.GL_BUFFER_UPDATE_BARRIER_BIT | GL_W.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	}

	@Override
	public void input(float dTime) {

	}

	@Override
	public void update(float dTime) {

	}

	@Override
	public void render(float dTime) {
		if (simConfig.time.realTime) {
			for (int i = 0; i < simConfig.time.frameSubSteps; i++) {
				compute();
				postProcess(dTime);
				if (iterationCount.increment() < simConfig.time.maxIterations()) {
					iterationFrame.increment();
				} else {
					readBack();
					break;
				}
			}
		}

//		System.err.println(iterationFrame.getValue() + " / " + iterationCount.getValue() + "/" + MAX_ITERATIONS);
		iterationFrame.set(0);

		GL_W.glViewport(0, 0, window.getWidth(), window.getHeight());
		GL_W.glClearColor(0.1f, 0.2f, 0.3f, 1f);
		GL_W.glClear(GL_W.GL_DEPTH_BUFFER_BIT | GL_W.GL_COLOR_BUFFER_BIT);

		GL_W.glEnable(GL_W.GL_DEPTH_TEST);
		GL_W.glDisable(GL_W.GL_CULL_FACE);

		scene.getCamera().getProjection().update(window.getSize());

		render(baseInstances);
		render(armInstances);
		render(headInstances);
		render(triangleEmitter);
	}

	private void render(InstanceEmitter emit) {
		instanceDirectShader.bind();
		emit.bind();
//		for (int i = 0; i < 4; i++) {
//			GL_W.glVertexAttribDivisor(InstanceEmitter.TRANSFORM_BUFFER_INDEX + i, 1);
//			GL_W.glEnableVertexAttribArray(InstanceEmitter.TRANSFORM_BUFFER_INDEX + i);
//		}

		instanceDirectShader.setUniform(DirectShader.HAS_TEXTURE, false);
		instanceDirectShader.setUniformUnsigned(DirectShader.INSTANCE_COUNT, instanceCount);
		instanceDirectShader.setUniform(RenderShader.TRANSFORMATION_MATRIX, GameEngine.IDENTITY_MATRIX4F);
		instanceDirectShader.setUniform(RenderShader.VIEW_MATRIX, scene.getCamera().getViewMatrix());
		instanceDirectShader.setUniform(RenderShader.PROJECTION_MATRIX, scene.getCamera().getProjection().getProjectionMatrix());

		GL_W.glDrawElementsInstanced(instanceDirectShader.getBeginMode()
				.getGlId(), emit.getParticleMesh().getIndicesCount(), GL_W.GL_UNSIGNED_INT, 0, emit.getParticleCount());

		emit.unbind();
	}

	@Override
	public void cleanup() {
		if (frame != null) {
			frame.dispatchEvent(new WindowEvent(frame, WindowEvent.WINDOW_CLOSING));
		}

		cleanup(pendulumBaseMesh, pendulumArmMesh, pendulumHeadMesh, triangleMesh);
		cleanup(baseInstances, armInstances, headInstances, triangleEmitter);
		cleanup(clearFloatComputeShader, clearVec4fComputeShader, clearMat4fComputeShader);
		cleanup(nnComputeComputeShader, nnPostprocessComputeShader);
		cleanup(directShader, instanceDirectShader);
		cleanup(weightsValueArray, biasesValueArray);
		cleanup(inputNeuronsValueArray, outputNeuronsValueArray, physicsVec4sValueArray);
		cleanup(baseTransformValueArray, armTransformValueArray, headTransformValueArray);
	}

	protected void cleanup(Cleanupable... vs) {
		for (Cleanupable v : vs) {
			if (v != null) {
				v.cleanup();
			}
		}
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
