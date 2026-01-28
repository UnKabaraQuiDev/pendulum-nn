package lu.kbra.pendulum_nn.shaders.compute.clear;

import org.joml.Vector3i;

import lu.kbra.standalone.gameengine.graph.shader.ComputeShader;
import lu.kbra.standalone.gameengine.graph.shader.part.ComputeShaderPart;

public class ClearVec4fComputeShader extends ComputeShader {

	public static final String ELEMENT_COUNT = "elementCount";

	public ClearVec4fComputeShader() {
		super(new ComputeShaderPart("classpath:/shaders/clear_vec4f.comp"), new Vector3i(256, 1, 1));
	}

	@Override
	public void createUniforms() {
		createUniform(ELEMENT_COUNT);
	}

}
