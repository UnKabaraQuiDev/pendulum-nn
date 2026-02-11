package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.AutoIncrement;
import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.ForeignKey;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.server.db.table.GenerationTable;

public class AgentPerformanceData implements DataBaseEntry {

	@Column
	@PrimaryKey
	@AutoIncrement
	protected long id;

	@Column
	@ForeignKey(table = AgentInstanceTable.class)
	protected int instanceHash;

	@Column
	@ForeignKey(table = GenerationTable.class)
	protected long generationId;

	@Column
	protected double score;

	public AgentPerformanceData() {
	}

	public AgentPerformanceData(long id) {
		this.id = id;
	}

	public AgentPerformanceData(int instanceHash, long generationId, double score) {
		this.instanceHash = instanceHash;
		this.generationId = generationId;
		this.score = score;
	}

	public long getId() {
		return id;
	}

	public int getInstanceHash() {
		return instanceHash;
	}

	public long getGenerationId() {
		return generationId;
	}

	public double getScore() {
		return score;
	}

	@Override
	public String toString() {
		return "AgentPerformanceData@" + System.identityHashCode(this) + " [id=" + id + ", instanceHash=" + instanceHash + ", generationId="
				+ generationId + ", score=" + score + "]";
	}

}
