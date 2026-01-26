package lu.kbra.pendulum_nn;

import lu.kbra.standalone.gameengine.cache.attrib.FloatAttribArray;
import lu.kbra.standalone.gameengine.generated.gl_wrapper.GL_W;
import lu.kbra.standalone.gameengine.utils.gl.consts.BufferType;

public class SyntheticFloatAttribArray extends FloatAttribArray {

	public SyntheticFloatAttribArray(String name, int index, int dataLength) {
		super(name, index, null);
		super.length = dataLength;
	}

	public SyntheticFloatAttribArray(String name, int index, int dataLength, BufferType bufferType) {
		super(name, index, null, bufferType);
		super.length = dataLength;
	}

	public SyntheticFloatAttribArray(String name, int index, int dataLength, BufferType bufferType, boolean _static) {
		super(name, index, null, bufferType, _static);
		super.length = dataLength;
	}

	public SyntheticFloatAttribArray(String name, int index, int dataLength, BufferType bufferType, boolean _static,
			int divisor) {
		super(name, index, null, bufferType, _static, divisor);
		super.length = dataLength;
	}

	public SyntheticFloatAttribArray(String name, int index, int dataLength, boolean _static) {
		super(name, index, null, _static);
		super.length = dataLength;
	}

	public SyntheticFloatAttribArray(String name, int index, int dataLength, boolean _static, int divisor) {
		super(name, index, null, _static, divisor);
		super.length = dataLength;
	}

	@Override
	public void init() {
		bind();

		GL_W.glBufferData(bufferType.getGlId(), length * getElementByteSize(), iStatic ? GL_W.GL_STATIC_DRAW : GL_W.GL_DYNAMIC_DRAW);

		if (bufferType == BufferType.ARRAY) {
			GL_W.glVertexAttribPointer(index, getElementComponentCount(), GL_W.GL_FLOAT, false, getElementByteSize(),
					0);
		}
	}

	@Override
	public void update() {
//		update(length);
		throw new UnsupportedOperationException("Synthetic arrays cannot be updated");
	}

	@Override
	public void update(float[] nPos) {
		if (iStatic) {
			throw new UnsupportedOperationException("Array is static.");
		} else if (nPos.length != super.length) {
			throw new IllegalArgumentException(
					"Use #resize to change the array's size (" + nPos.length + "<>" + super.length + ").");
		}
		bind();

		GL_W.glBufferSubData(bufferType.getGlId(), 0, nPos);
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

		if (isVertexArray()) {
			GL_W.glVertexAttribPointer(index, getElementComponentCount(), GL_W.GL_FLOAT, false, getElementByteSize(),
					0);
		}
	}

	@Override
	public void resize(float[] nPos) {
		bind();

		if (nPos.length == super.length) {
			GL_W.glBufferSubData(bufferType.getGlId(), 0, nPos);
		} else {
			GL_W.glBufferData(bufferType.getGlId(), nPos, iStatic ? GL_W.GL_STATIC_DRAW : GL_W.GL_DYNAMIC_DRAW);
		}

		super.length = nPos.length;

		if (isVertexArray()) {
			GL_W.glVertexAttribPointer(index, getElementComponentCount(), GL_W.GL_FLOAT, false, getElementByteSize(),
					0);
		}
	}

	@Override
	public boolean isLoaded() {
		return false;
	}

}
