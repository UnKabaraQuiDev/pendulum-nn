package lu.kbra.pendulum_nn.shaders.compute;

import java.util.Map;

import org.joml.Vector3i;
import org.joml.Vector3ic;

import lu.kbra.standalone.gameengine.graph.shader.ComputeShader;
import lu.kbra.standalone.gameengine.graph.shader.part.ComputeShaderPart;

public class NNPostprocessComputeShader extends ComputeShader {

	public static Vector3ic LOCAL_SIZE = new Vector3i(64, 0, 0);

	public static final String D_TIME = "dTime";
//	public static final String INPUT_SIZE = "inputSize";
	public static final String INSTANCE_COUNT = "instanceCount";
	public static final String GRAVITY = "gravity";
	public static final String PENDULUM_LENGTH = "pendulumLength";
	public static final String BOUNDS = "bounds";
	public static final String FRICTION = "friction";
	public static final String ANGULAR_FRICTION = "angularFriction";
	public static final String ACCELERATION_BOUNDS = "accBounds";
	public static final String ACCELERATION_FACTOR = "accFactor";
	public static final String USER_FORCE_DIR = "userForceDir";
	public static final String USER_FORCE_SOURCE = "userForceSource";
	public static final String HEAD_RADIUS = "headRadius"; // 0.14
	public static final String MASS = "mass";

	public NNPostprocessComputeShader() {
		super(new ComputeShaderPart("classpath:/shaders/nn_postprocess.comp", getBuildingDeps()), LOCAL_SIZE);
	}

	private static Map<String, Object> getBuildingDeps() {
		final Map<String, Object> buildingDeps = getBaseBuildingDeps(LOCAL_SIZE);
		buildingDeps.put("%PI%", Math.PI);
		return buildingDeps;
	}

	@Override
	public void createUniforms() {
		createUniform(D_TIME);
//		createUniform(INPUT_SIZE);
		createUniform(INSTANCE_COUNT);
		createUniform(GRAVITY);
		createUniform(PENDULUM_LENGTH);
		createUniform(BOUNDS);
		createUniform(FRICTION);
		createUniform(ANGULAR_FRICTION);
		createUniform(ACCELERATION_BOUNDS);
		createUniform(ACCELERATION_FACTOR);
		createUniform(USER_FORCE_DIR);
		createUniform(USER_FORCE_SOURCE);
		createUniform(HEAD_RADIUS);
		createUniform(MASS);
	}

}
