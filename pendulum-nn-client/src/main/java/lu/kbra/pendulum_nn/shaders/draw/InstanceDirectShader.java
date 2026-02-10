package lu.kbra.pendulum_nn.shaders.draw;

import lu.kbra.standalone.gameengine.graph.shader.part.AbstractShaderPart;

public final class InstanceDirectShader extends DirectShader {

	public InstanceDirectShader() {
		super(AbstractShaderPart.load("classpath:/shaders/gbuffer_inst.vert"), AbstractShaderPart.load("classpath:/shaders/direct.frag"));
	}

}
