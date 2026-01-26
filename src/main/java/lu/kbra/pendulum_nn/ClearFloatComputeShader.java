package lu.kbra.pendulum_nn;

import org.joml.Vector3i;

import lu.kbra.standalone.gameengine.graph.shader.ComputeShader;
import lu.kbra.standalone.gameengine.graph.shader.part.ComputeShaderPart;

public class ClearFloatComputeShader extends ComputeShader {

	public static final String ELEMENT_COUNT = "elementCount";

	public ClearFloatComputeShader() {
		super(new ComputeShaderPart("classpath:/shaders/clear_float.comp"), new Vector3i(256, 1, 1));
	}

	@Override
	public void createUniforms() {
		createUniform(ELEMENT_COUNT);
	}

}
