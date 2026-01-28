package lu.kbra.pendulum_nn.shaders.compute;

import java.util.Map;

import org.joml.Vector3i;
import org.joml.Vector3ic;

import lu.kbra.pendulum_nn.ActivationFunction;
import lu.kbra.standalone.gameengine.graph.shader.ComputeShader;
import lu.kbra.standalone.gameengine.graph.shader.part.ComputeShaderPart;

public class NNComputeComputeShader extends ComputeShader {

	public static int MAX_LAYERS = 8;
	public static int MAX_NEURONS = 64;

	public static Vector3ic LOCAL_SIZE = new Vector3i(32, 32, 32);

	public static final String LAYER_COUNT = "layerCount";
	public static final String LAYER_SIZE = "layerSize";
	public static final String WEIGHT_OFFSET_PER_INSTANCE = "weightOffsetPerInstance";
	public static final String BIAS_OFFSET_PER_INSTANCE = "biasOffsetPerInstance";
	public static final String INSTANCE_COUNT = "instanceCount";
	public static final String ACTIVATION_FUNCTION = "activationFunction";

	public NNComputeComputeShader() {
		super(new ComputeShaderPart("classpath:/shaders/nn_compute.comp", getBuildingDeps()), LOCAL_SIZE);
	}

	private static Map<String, Object> getBuildingDeps() {
		final Map<String, Object> buildingDeps = getBaseBuildingDeps(LOCAL_SIZE);
		buildingDeps.put("%MAX_LAYERS%", MAX_LAYERS);
		buildingDeps.put("%MAX_NEURONS%", MAX_NEURONS);

		for (ActivationFunction af : ActivationFunction.values()) {
			buildingDeps.put("%" + af.name() + "%", af.ordinal());
		}

		return buildingDeps;
	}

	@Override
	public void createUniforms() {
		createUniform(INPUT_SIZE);
		createUniform(LAYER_COUNT);
		createUniform(LAYER_SIZE);
		createUniform(WEIGHT_OFFSET_PER_INSTANCE);
		createUniform(BIAS_OFFSET_PER_INSTANCE);
		createUniform(INSTANCE_COUNT);
		createUniform(ACTIVATION_FUNCTION);
	}

}
