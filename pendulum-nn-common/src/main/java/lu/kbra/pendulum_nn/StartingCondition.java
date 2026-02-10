package lu.kbra.pendulum_nn;

public class StartingCondition implements JacksonObject {

	private float xPosition;
	private float angle;
	private float velocity;
	private float angularVelocity;

	public StartingCondition() {
	}

	public StartingCondition(float xPosition, float angle, float velocity, float angularVelocity) {
		this.xPosition = xPosition;
		this.angle = angle;
		this.velocity = velocity;
		this.angularVelocity = angularVelocity;
	}

	public float getxPosition() {
		return xPosition;
	}

	public void setxPosition(float xPosition) {
		this.xPosition = xPosition;
	}

	public float getAngle() {
		return angle;
	}

	public void setAngle(float angle) {
		this.angle = angle;
	}

	public float getVelocity() {
		return velocity;
	}

	public void setVelocity(float velocity) {
		this.velocity = velocity;
	}

	public float getAngularVelocity() {
		return angularVelocity;
	}

	public void setAngularVelocity(float angularVelocity) {
		this.angularVelocity = angularVelocity;
	}

	@Override
	public String toString() {
		return "StartingCondition@" + System.identityHashCode(this) + " [xPosition=" + xPosition + ", angle=" + angle
				+ ", velocity=" + velocity + ", angularVelocity=" + angularVelocity + "]";
	}

}
