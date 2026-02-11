package lu.kbra.pendulum_nn.server.db.data;

import lu.kbra.pclib.db.autobuild.column.Column;
import lu.kbra.pclib.db.autobuild.column.PrimaryKey;
import lu.kbra.pclib.db.impl.DataBaseEntry;
import lu.kbra.pendulum_nn.SimulationConfiguration;

public class SimConfigData implements DataBaseEntry {

	@Column
	@PrimaryKey
	protected int hash;

	@Column
	protected SimulationConfiguration config;

	public SimConfigData() {
	}

	public SimConfigData(int hash) {
		this.hash = hash;
	}

	public SimConfigData(SimulationConfiguration config) {
		this.hash = config.hashCode();
		this.config = config;
	}

	public int getHash() {
		return hash;
	}

	public SimulationConfiguration getConfig() {
		return config;
	}

	@Override
	public int hashCode() {
		return hash;
	}

	@Override
	public String toString() {
		return "SimConfigData@" + System.identityHashCode(this) + " [hash=" + hash + ", config=" + config + "]";
	}

}
