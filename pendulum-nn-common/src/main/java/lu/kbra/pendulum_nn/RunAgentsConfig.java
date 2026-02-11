package lu.kbra.pendulum_nn;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class RunAgentsConfig {

	protected final boolean random;
	protected final NNStructure structure;
	protected final List<NNInstance> instances;
	protected final int totalCount;
	protected final int totalGenerations;

	@JsonCreator
	public RunAgentsConfig(@JsonProperty("random") boolean random, @JsonProperty("structure") NNStructure structure,
			@JsonProperty("instances") List<NNInstance> instances, @JsonProperty("totalCount") int totalCount,
			@JsonProperty("totalGenerations") int totalGenerations) {
		this.random = random;
		this.structure = structure;
		this.instances = instances;
		this.totalCount = totalCount;
		this.totalGenerations = totalGenerations;
	}

	public boolean isRandom() {
		return random;
	}

	public NNStructure getStructure() {
		return structure;
	}

	public List<NNInstance> getInstances() {
		return instances;
	}

	public int getTotalCount() {
		return totalCount;
	}

	public int getTotalGenerations() {
		return totalGenerations;
	}

	@Override
	public String toString() {
		return "RunAgentsConfig@" + System.identityHashCode(this) + " [random=" + random + ", structure=" + structure
				+ ", instances=" + instances + ", totalCount=" + totalCount + ", totalGenerations=" + totalGenerations
				+ "]";
	}

}
