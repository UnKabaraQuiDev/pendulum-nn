package lu.kbra.pendulum_nn;

import java.util.List;

import lu.kbra.pclib.datastructure.pair.ReadOnlyPair;

public class GenerationReport {

	private final long simulationId;
	private final int genIndex;
	private final List<ReadOnlyPair<Float, NNInstance>> topAgents;
	private final StartingCondition startingCondition;
	private final SimulationConfiguration simulationConfiguration;
	private final NNStructure structure;

	public GenerationReport(long simulationId, int genIndex, List<ReadOnlyPair<Float, NNInstance>> topAgents,
			StartingCondition startingCondition, SimulationConfiguration simulationConfiguration,
			NNStructure structure) {
		this.simulationId = simulationId;
		this.genIndex = genIndex;
		this.topAgents = topAgents;
		this.startingCondition = startingCondition;
		this.simulationConfiguration = simulationConfiguration;
		this.structure = structure;
	}

	public long getSimulationId() {
		return this.simulationId;
	}

	public int getGenerationIndex() {
		return this.genIndex;
	}

	public List<ReadOnlyPair<Float, NNInstance>> getTopAgents() {
		return this.topAgents;
	}

	public StartingCondition getStartingCondition() {
		return startingCondition;
	}

	public SimulationConfiguration getSimulationConfiguration() {
		return simulationConfiguration;
	}

	public NNStructure getStructure() {
		return structure;
	}

	@Override
	public String toString() {
		return "GenerationReport@" + System.identityHashCode(this) + " [simulationId=" + simulationId + ", genIndex="
				+ genIndex + ", topAgents=" + topAgents + ", startingCondition=" + startingCondition
				+ ", simulationConfiguration=" + simulationConfiguration + ", structure=" + structure + "]";
	}

}
