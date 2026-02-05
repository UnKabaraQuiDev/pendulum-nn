package lu.kbra.pendulum_nn;

import org.joml.Math;

public class SimulationConfiguration {

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
		public boolean realTime = false;
		public int frameSubSteps = 10;

		public int maxIterations() {
			return ups * virtualSeconds;
		}

		@Override
		public String toString() {
			return "Time@" + System.identityHashCode(this) + " [virtualSeconds=" + virtualSeconds + ", ups=" + ups + ", fixedDTime="
					+ fixedDTime + ", realTime=" + realTime + ", frameSubSteps=" + frameSubSteps + ", maxIterations()=" + maxIterations()
					+ "]";
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
		public String toString() {
			return "Physics@" + System.identityHashCode(this) + " [gravity=" + gravity + ", mass=" + mass + ", angularFriction="
					+ angularFriction + ", friction=" + friction + ", pendulumLength=" + pendulumLength + ", accelerationFactor="
					+ accelerationFactor + ", accelerationMax=" + accelerationMax + ", width=" + width + "]";
		}

	}

	public static class NeuralNetworks {

		public float mutateRate = 0.5f;
		public float mutateStrength = 1f;
		public int topAgents = 100;
		public int totalAgents = 1024 * 10;

		@Override
		public String toString() {
			return "NeuralNetworks@" + System.identityHashCode(this) + " [mutateRate=" + mutateRate + ", mutateStrength=" + mutateStrength
					+ ", topAgents=" + topAgents + ", totalAgents=" + totalAgents + "]";
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
		public String toString() {
			return "Persistance@" + System.identityHashCode(this) + " [reloadLatest=" + reloadLatest + ", saveDir=" + saveDir
					+ ", reloadSpecific=" + reloadSpecific + ", reloadFile=" + reloadFile + "]";
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
		public String toString() {
			return "StartState@" + System.identityHashCode(this) + " [fixedSimulationParameters=" + fixedSimulationParameters
					+ ", startingPosition=" + startingPosition + ", startingVelocity=" + startingVelocity + ", startingAngle="
					+ startingAngle + ", startingAngleVelocity=" + startingAngleVelocity + ", startingPositionScale="
					+ startingPositionScale + ", startingAngleScale=" + startingAngleScale + ", startingVelocityScale="
					+ startingVelocityScale + ", startingAngleVelocityScale=" + startingAngleVelocityScale + "]";
		}

	}

	@Override
	public String toString() {
		return "SimulationConfiguration@" + System.identityHashCode(this) + " [hotReloadShaders=" + hotReloadShaders
				+ ", debugPerfectScore=" + debugPerfectScore + ", time=" + time + ", physics=" + physics + ", neuralNetworks="
				+ neuralNetworks + ", persistance=" + persistance + ", startState=" + startState + "]";
	}

}
