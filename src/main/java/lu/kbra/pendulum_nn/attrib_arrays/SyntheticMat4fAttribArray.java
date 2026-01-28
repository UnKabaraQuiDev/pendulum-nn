package lu.kbra.pendulum_nn.attrib_arrays;

import org.joml.Matrix4f;

import lu.kbra.standalone.gameengine.cache.attrib.Mat4fAttribArray;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.utils.GameEngineUtils;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

public class SyntheticMat4fAttribArray extends Mat4fAttribArray {

	public SyntheticMat4fAttribArray(String name, int index, int dataLength, boolean iStatic, int divisor) {
		super(name, index, null, iStatic, divisor);
		super.length = dataLength;
	}

	public SyntheticMat4fAttribArray(String name, int index, int dataLength, boolean iStatic) {
		super(name, index, null, iStatic);
		super.length = dataLength;
	}

	public SyntheticMat4fAttribArray(String name, int index, int dataLength, BufferType bufferType, boolean iStatic, int divisor) {
		super(name, index, null, bufferType, iStatic, divisor);
		super.length = dataLength;
	}

	public SyntheticMat4fAttribArray(String name, int index, int dataLength, BufferType bufferType, boolean s) {
		super(name, index, null, bufferType, s);
		super.length = dataLength;
	}

	public SyntheticMat4fAttribArray(String name, int index, int dataLength, BufferType bufferType) {
		super(name, index, null, bufferType);
		super.length = dataLength;
	}

	public SyntheticMat4fAttribArray(String name, int index, int dataLength) {
		super(name, index, null);
		super.length = dataLength;
	}

	@Override
	public void init() {
		bind();

		GL_W.glBufferData(bufferType.getGlId(), length * getElementByteSize(), iStatic ? GL_W.GL_STATIC_DRAW : GL_W.GL_DYNAMIC_DRAW);
	}

	@Override
	public void update() {
//		update(length);
		throw new UnsupportedOperationException("Synthetic arrays cannot be updated");
	}

	@Override
	public void update(Matrix4f[] nPos) {
		if (iStatic) {
			throw new UnsupportedOperationException("Array is static.");
		} else if (nPos.length != super.length) {
			throw new IllegalArgumentException("Use #resize to change the array's size (" + nPos.length + "<>" + super.length + ").");
		}
		bind();

		GL_W.glBufferSubData(bufferType.getGlId(), 0, GameEngineUtils.toFlatArray(nPos));
	}

	public void resize(int newSize) {
		bind();

		if (newSize == super.length) {
//			GL_W.glBufferSubData(bufferType.getGlId(), 0, nPos);
			return;
		} else {
			GL_W.glBufferData(bufferType.getGlId(), newSize * getElementByteSize(), iStatic ? GL_W.GL_STATIC_DRAW : GL_W.GL_DYNAMIC_DRAW);
		}

		super.length = newSize;
	}

	@Override
	public void resize(Matrix4f[] nPos) {
		bind();

		if (nPos.length == super.length) {
			GL_W.glBufferSubData(bufferType.getGlId(), 0, GameEngineUtils.toFlatArray(nPos));
		} else {
			GL_W.glBufferData(bufferType.getGlId(),
					GameEngineUtils.toFlatArray(nPos),
					iStatic ? GL_W.GL_STATIC_DRAW : GL_W.GL_DYNAMIC_DRAW);
		}

		super.length = nPos.length;
	}

	@Override
	public boolean isLoaded() {
		return false;
	}

}
