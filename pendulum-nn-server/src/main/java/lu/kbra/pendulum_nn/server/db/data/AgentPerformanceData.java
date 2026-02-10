package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.AutoIncrement;
import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.ForeignKey;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.NNInstance;
import lu.kbra.pendulum_nn.server.db.table.GenerationTable;

public class AgentPerformanceData implements DataBaseEntry {

	@Column
	@PrimaryKey
	@AutoIncrement
	protected long id;

	@Column
	protected NNInstance instance;

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

	public AgentPerformanceData(NNInstance instance, long generationId, double score) {
		this.instance = instance;
		this.generationId = generationId;
		this.score = score;
	}

	public long getId() {
		return id;
	}

	public NNInstance getInstance() {
		return instance;
	}

	public long getGenerationId() {
		return generationId;
	}

	public double getScore() {
		return score;
	}

	@Override
	public String toString() {
		return "AgentPerformanceData@" + System.identityHashCode(this) + " [id=" + id + ", instance=" + instance
				+ ", generationId=" + generationId + ", score=" + score + "]";
	}

}
