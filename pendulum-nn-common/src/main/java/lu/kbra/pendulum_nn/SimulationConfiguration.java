package lu.kbra.pendulum_nn;

import java.util.Objects;

public class SimulationConfiguration implements JacksonObject {

	public boolean hotReloadShaders = false;
	public boolean debugPerfectScore = false;

	public Time time;
	public Physics physics;
	public NeuralNetworks neuralNetworks;
	public Persistance persistance;
	public StartState startState;

	public SimulationConfiguration() {
	}

	public SimulationConfiguration(final boolean defaults) {
		if (!defaults) {
			return;
		}
		time = new Time();
		physics = new Physics();
		neuralNetworks = new NeuralNetworks();
		persistance = new Persistance();
		startState = new StartState();
	}

	public static class Time {

		public int virtualSeconds = 300;
		public int ups = 60;
		public float fixedDTime = 1f / ups;
//		public boolean realTime = false;
		public int frameSubSteps = 10;

		public int maxIterations() {
			return ups * virtualSeconds;
		}

		@Override
		public int hashCode() {
			return Objects.hash(fixedDTime, frameSubSteps, ups, virtualSeconds);
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Time other = (Time) obj;
			return Float.floatToIntBits(fixedDTime) == Float.floatToIntBits(other.fixedDTime)
					&& frameSubSteps == other.frameSubSteps && ups == other.ups
					&& virtualSeconds == other.virtualSeconds;
		}

		@Override
		public String toString() {
			return "Time@" + System.identityHashCode(this) + " [virtualSeconds=" + virtualSeconds + ", ups=" + ups
					+ ", fixedDTime=" + fixedDTime + ", frameSubSteps=" + frameSubSteps + "]";
		}

	}

	public static class Physics {

		public float gravity = 9f;
		public float mass = 10f;
		public float angularFriction = 0.4f;
		public float friction = 0.25f;
		public float pendulumLength = 1f;
		public float accelerationFactor = 10f;
		public float accelerationMax = 5;
		public float width = 3f;

		@Override
		public int hashCode() {
			return Objects.hash(accelerationFactor, accelerationMax, angularFriction, friction, gravity, mass,
					pendulumLength, width);
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Physics other = (Physics) obj;
			return Float.floatToIntBits(accelerationFactor) == Float.floatToIntBits(other.accelerationFactor)
					&& Float.floatToIntBits(accelerationMax) == Float.floatToIntBits(other.accelerationMax)
					&& Float.floatToIntBits(angularFriction) == Float.floatToIntBits(other.angularFriction)
					&& Float.floatToIntBits(friction) == Float.floatToIntBits(other.friction)
					&& Float.floatToIntBits(gravity) == Float.floatToIntBits(other.gravity)
					&& Float.floatToIntBits(mass) == Float.floatToIntBits(other.mass)
					&& Float.floatToIntBits(pendulumLength) == Float.floatToIntBits(other.pendulumLength)
					&& Float.floatToIntBits(width) == Float.floatToIntBits(other.width);
		}

		@Override
		public String toString() {
			return "Physics@" + System.identityHashCode(this) + " [gravity=" + gravity + ", mass=" + mass
					+ ", angularFriction=" + angularFriction + ", friction=" + friction + ", pendulumLength="
					+ pendulumLength + ", accelerationFactor=" + accelerationFactor + ", accelerationMax="
					+ accelerationMax + ", width=" + width + "]";
		}

	}

	public static class NeuralNetworks {

		public float mutateRate = 0.5f;
		public float mutateStrength = 1f;
		public int topAgents = 100;
		public int totalAgents = 1024 * 10;

		@Override
		public int hashCode() {
			return Objects.hash(mutateRate, mutateStrength, topAgents, totalAgents);
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			NeuralNetworks other = (NeuralNetworks) obj;
			return Float.floatToIntBits(mutateRate) == Float.floatToIntBits(other.mutateRate)
					&& Float.floatToIntBits(mutateStrength) == Float.floatToIntBits(other.mutateStrength)
					&& topAgents == other.topAgents && totalAgents == other.totalAgents;
		}

		@Override
		public String toString() {
			return "NeuralNetworks@" + System.identityHashCode(this) + " [mutateRate=" + mutateRate
					+ ", mutateStrength=" + mutateStrength + ", topAgents=" + topAgents + ", totalAgents=" + totalAgents
					+ "]";
		}

	}

	public static class Persistance {

		public boolean reloadLatest = false;
		public String reloadDir = "./output/";
		public boolean save = true;
		public int limitSave = 0;
		public String saveDir = "./output/";
		public boolean reloadSpecific = false;
		public String reloadFile = "./output/";

		@Override
		public int hashCode() {
			return Objects.hash(limitSave, reloadDir, reloadFile, reloadLatest, reloadSpecific, save, saveDir);
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Persistance other = (Persistance) obj;
			return limitSave == other.limitSave && Objects.equals(reloadDir, other.reloadDir)
					&& Objects.equals(reloadFile, other.reloadFile) && reloadLatest == other.reloadLatest
					&& reloadSpecific == other.reloadSpecific && save == other.save
					&& Objects.equals(saveDir, other.saveDir);
		}

		@Override
		public String toString() {
			return "Persistance@" + System.identityHashCode(this) + " [reloadLatest=" + reloadLatest + ", saveDir="
					+ saveDir + ", reloadSpecific=" + reloadSpecific + ", reloadFile=" + reloadFile + "]";
		}

	}

	public static class StartState {

		public boolean fixedSimulationParameters = true;

		public float startingPosition = 0;
		public float startingVelocity = 0;
		public float startingAngle = (float) Math.PI / 100 * 60;
		public float startingAngleVelocity = 0;

		public float startingPositionScale = 0.02f;
		public float startingAngleScale = 0.02f;
		public float startingVelocityScale = 0.05f;
		public float startingAngleVelocityScale = 0.05f;

		@Override
		public int hashCode() {
			return Objects.hash(fixedSimulationParameters, startingAngle, startingAngleScale, startingAngleVelocity,
					startingAngleVelocityScale, startingPosition, startingPositionScale, startingVelocity,
					startingVelocityScale);
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			StartState other = (StartState) obj;
			return fixedSimulationParameters == other.fixedSimulationParameters
					&& Float.floatToIntBits(startingAngle) == Float.floatToIntBits(other.startingAngle)
					&& Float.floatToIntBits(startingAngleScale) == Float.floatToIntBits(other.startingAngleScale)
					&& Float.floatToIntBits(startingAngleVelocity) == Float.floatToIntBits(other.startingAngleVelocity)
					&& Float.floatToIntBits(startingAngleVelocityScale) == Float
							.floatToIntBits(other.startingAngleVelocityScale)
					&& Float.floatToIntBits(startingPosition) == Float.floatToIntBits(other.startingPosition)
					&& Float.floatToIntBits(startingPositionScale) == Float.floatToIntBits(other.startingPositionScale)
					&& Float.floatToIntBits(startingVelocity) == Float.floatToIntBits(other.startingVelocity)
					&& Float.floatToIntBits(startingVelocityScale) == Float.floatToIntBits(other.startingVelocityScale);
		}

		@Override
		public String toString() {
			return "StartState@" + System.identityHashCode(this) + " [fixedSimulationParameters="
					+ fixedSimulationParameters + ", startingPosition=" + startingPosition + ", startingVelocity="
					+ startingVelocity + ", startingAngle=" + startingAngle + ", startingAngleVelocity="
					+ startingAngleVelocity + ", startingPositionScale=" + startingPositionScale
					+ ", startingAngleScale=" + startingAngleScale + ", startingVelocityScale=" + startingVelocityScale
					+ ", startingAngleVelocityScale=" + startingAngleVelocityScale + "]";
		}

	}

	@Override
	public int hashCode() {
		return Objects.hash(debugPerfectScore, hotReloadShaders, neuralNetworks, persistance, physics, startState,
				time);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		SimulationConfiguration other = (SimulationConfiguration) obj;
		return debugPerfectScore == other.debugPerfectScore && hotReloadShaders == other.hotReloadShaders
				&& Objects.equals(neuralNetworks, other.neuralNetworks)
				&& Objects.equals(persistance, other.persistance) && Objects.equals(physics, other.physics)
				&& Objects.equals(startState, other.startState) && Objects.equals(time, other.time);
	}

	@Override
	public String toString() {
		return "SimulationConfiguration@" + System.identityHashCode(this) + " [hotReloadShaders=" + hotReloadShaders
				+ ", debugPerfectScore=" + debugPerfectScore + ", time=" + time + ", physics=" + physics
				+ ", neuralNetworks=" + neuralNetworks + ", persistance=" + persistance + ", startState=" + startState
				+ "]";
	}

}
