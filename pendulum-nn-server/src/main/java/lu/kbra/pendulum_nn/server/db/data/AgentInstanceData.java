package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.NNInstance;

public class AgentInstanceData implements DataBaseEntry {

	@Column
	@PrimaryKey
	protected int hash;

	@Column
	protected NNInstance instance;

	public AgentInstanceData() {
	}

	public AgentInstanceData(int hash) {
		this.hash = hash;
	}

	public AgentInstanceData(NNInstance instance) {
		this.hash = instance.hashCode();
		this.instance = instance;
	}

	public int getHash() {
		return hash;
	}

	public NNInstance getInstance() {
		return instance;
	}

	@Override
	public String toString() {
		return "AgentInstanceData@" + System.identityHashCode(this) + " [hash=" + hash + ", instance=" + instance + "]";
	}

}
