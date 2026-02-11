package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry.ReadOnlyDataBaseEntry;

public class ROAgentStructureScoreData implements ReadOnlyDataBaseEntry {

	@PrimaryKey
	@Column
	protected int structureHash;

	@Column
	protected double averageScore;

	public int getStructureHash() {
		return structureHash;
	}

	public double getAverageScore() {
		return averageScore;
	}

	@Override
	public String toString() {
		return "ROAgentStructureScoreData@" + System.identityHashCode(this) + " [structureHash=" + structureHash + ", averageScore="
				+ averageScore + "]";
	}

}
