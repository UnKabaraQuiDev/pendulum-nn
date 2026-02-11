package lu.kbra.pendulum_nn.renderable;

import java.util.function.IntFunction;

import lu.kbra.standalone.gameengine.cache.attrib.Mat4fAttribArray;
import lu.kbra.standalone.gameengine.cache.attrib.impl.AttribArray;
import lu.kbra.standalone.gameengine.geom.Mesh;
import lu.kbra.standalone.gameengine.geom.instance.InstanceEmitter;
import lu.kbra.standalone.gameengine.utils.transform.Transform;

public class LimitedInstanceEmitter extends InstanceEmitter {

	protected int particleCount;

	public LimitedInstanceEmitter(String name, Mesh mesh, int count, IntFunction<Transform> baseTransform, AttribArray... attribs) {
		super(name, mesh, count, baseTransform, attribs);
		this.particleCount = count;
	}

	public LimitedInstanceEmitter(String name, Mesh mesh, int count, Transform baseTransform, AttribArray... attribs) {
		super(name, mesh, count, baseTransform, attribs);
		this.particleCount = count;
	}

	public LimitedInstanceEmitter(String name, Mesh mesh, int count, Mat4fAttribArray baseTransform, AttribArray... attribs) {
		super(name, mesh, count, baseTransform, attribs);
		this.particleCount = count;
	}

	@Override
	public int getParticleCount() {
		return particleCount;
	}

	public void setParticleCount(int particleCount) {
		this.particleCount = particleCount;
	}

}
