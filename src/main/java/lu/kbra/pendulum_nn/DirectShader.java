package lu.kbra.pendulum_nn;

import org.joml.Vector4f;

import lu.kbra.standalone.gameengine.graph.shader.RenderShader;
import lu.kbra.standalone.gameengine.graph.shader.part.AbstractShaderPart;

public sealed class DirectShader extends RenderShader permits InstanceDirectShader {

	public static final String TXT0 = "txt0";
	public static final String INSTANCE_COUNT = "instanceCount";
	public static final String HAS_TEXTURE = "hasTexture";

	public static final Vector4f DEFAULT_TINT = new Vector4f(1);

	public DirectShader() {
		super(AbstractShaderPart.load("classpath:/shaders/gbuffer.vert"), AbstractShaderPart.load("classpath:/shaders/direct.frag"));
	}

	DirectShader(final AbstractShaderPart... parts) {
		super(parts);
	}

	@Override
	public void createUniforms() {
		super.createSceneUniforms();

		super.createUniform(TXT0);
		super.createUniform(INSTANCE_COUNT);
		super.createUniform(HAS_TEXTURE);
	}

}
