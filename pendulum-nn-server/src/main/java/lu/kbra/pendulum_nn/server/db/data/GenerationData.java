package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.AutoIncrement;
import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.ForeignKey;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.server.db.table.AgentStructureTable;
import lu.kbra.pendulum_nn.server.db.table.SimConfigTable;
import lu.kbra.pendulum_nn.server.db.table.StartingConditionTable;

public class GenerationData implements DataBaseEntry {

	@Column
	@PrimaryKey
	@AutoIncrement
	protected long id;

	@Column
	protected int generation;

	@Column
	@ForeignKey(table = AgentStructureTable.class)
	protected int structureHash;

	@Column
	@ForeignKey(table = SimConfigTable.class)
	protected int simConfigHash;

	@Column
	@ForeignKey(table = StartingConditionTable.class)
	protected int startingConditionHash;

	public GenerationData() {
	}

	public GenerationData(long id) {
		this.id = id;
	}

	public GenerationData(int generation, int structureHash, int simConfigHash, int startingConditionHash) {
		this.generation = generation;
		this.structureHash = structureHash;
		this.simConfigHash = simConfigHash;
		this.startingConditionHash = startingConditionHash;
	}

	public long getId() {
		return id;
	}

	public int getGeneration() {
		return generation;
	}

	public int getStructureHash() {
		return structureHash;
	}

	public int getSimConfigHash() {
		return simConfigHash;
	}

	public int getStartingConditionHash() {
		return startingConditionHash;
	}

	@Override
	public String toString() {
		return "GenerationData@" + System.identityHashCode(this) + " [id=" + id + ", generation=" + generation
				+ ", structureHash=" + structureHash + ", simConfigHash=" + simConfigHash + ", startingConditionHash="
				+ startingConditionHash + "]";
	}

}
