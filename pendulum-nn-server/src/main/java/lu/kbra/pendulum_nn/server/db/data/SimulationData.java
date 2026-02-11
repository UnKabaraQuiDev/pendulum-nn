package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.AutoIncrement;
import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.ForeignKey;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.server.db.table.AgentStructureTable;
import lu.kbra.pendulum_nn.server.db.table.SimConfigTable;

public class SimulationData implements DataBaseEntry {

	@Column
	@PrimaryKey
	@AutoIncrement
	protected long id;

	@Column
	@ForeignKey(table = SimConfigTable.class)
	protected int simulationHash;

	@Column
	@ForeignKey(table = AgentStructureTable.class)
	protected int structureHash;

	public SimulationData() {
	}

	public SimulationData(long id) {
		this.id = id;
	}

	public SimulationData(int simulationHash, int structureHash) {
		this.simulationHash = simulationHash;
		this.structureHash = structureHash;
	}

	public long getId() {
		return id;
	}

	public int getSimulationHash() {
		return simulationHash;
	}

	public int getStructureHash() {
		return structureHash;
	}

	@Override
	public String toString() {
		return "SimulationData@" + System.identityHashCode(this) + " [id=" + id + ", simulationHash=" + simulationHash + ", structureHash="
				+ structureHash + "]";
	}

}
