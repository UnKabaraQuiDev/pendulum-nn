package lu.kbra.pendulum_nn.shaders.compute.fill;

import java.util.HashMap;
import java.util.Map;

import org.joml.Vector3i;

import lu.kbra.standalone.gameengine.graph.shader.ComputeShader;
import lu.kbra.standalone.gameengine.graph.shader.part.ComputeShaderPart;

public class FillVec4fComputeShader extends ComputeShader {

	public static final String ELEMENT_COUNT = "elementCount";
	public static final String DEFAULT_COUNT = "defaultCount";
	public static final String DEFAULTS = "defaults";

	public FillVec4fComputeShader() {
		super(new ComputeShaderPart("classpath:/shaders/fill_vec4f.comp", getBuildingDeps()), new Vector3i(256, 1, 1));
	}

	private static Map<String, Object> getBuildingDeps() {
		return new HashMap<String, Object>() {
			{
				put("%MAX_DEFAULTS%", 16);
			}
		};
	}

	@Override
	public void createUniforms() {
		createUniform(ELEMENT_COUNT);
		createUniform(DEFAULT_COUNT);
		createUniform(DEFAULTS);
	}

}
