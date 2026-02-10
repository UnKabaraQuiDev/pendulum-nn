package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.StartingCondition;

public class StartingConditionData implements DataBaseEntry {

	@Column
	@PrimaryKey
	protected int hash;

	@Column
	protected StartingCondition startingCondition;

	public StartingConditionData() {
	}

	public StartingConditionData(int hash) {
		this.hash = hash;
	}

	public StartingConditionData(StartingCondition startingCondition) {
		this.hash = startingCondition.hashCode();
		this.startingCondition = startingCondition;
	}

	public int getHash() {
		return hash;
	}

	public StartingCondition getStartingCondition() {
		return startingCondition;
	}

	@Override
	public int hashCode() {
		return hash;
	}

	@Override
	public String toString() {
		return "StartingConditionData@" + System.identityHashCode(this) + " [hash=" + hash + ", startingCondition="
				+ startingCondition + "]";
	}

}
