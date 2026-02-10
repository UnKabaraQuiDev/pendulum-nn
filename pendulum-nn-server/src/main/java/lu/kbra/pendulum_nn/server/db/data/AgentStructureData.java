package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.NNStructure;

public class AgentStructureData implements DataBaseEntry {

	@Column
	@PrimaryKey
	protected int hash;

	@Column
	protected NNStructure structure;

	public AgentStructureData() {
	}

	public AgentStructureData(int hash) {
		this.hash = hash;
	}

	public AgentStructureData(NNStructure structure) {
		this.structure = structure;
	}

	public int getHash() {
		return hash;
	}

	public NNStructure getStructure() {
		return structure;
	}

	@Override
	public int hashCode() {
		return hash;
	}

	@Override
	public String toString() {
		return "AgentStructureData@" + System.identityHashCode(this) + " [hash=" + hash + ", structure=" + structure
				+ "]";
	}

}
