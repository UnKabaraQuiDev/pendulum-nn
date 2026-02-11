package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.SimulationData;

@Component
public abstract class SimulationTable extends DeferredDataBaseTable<SimulationData> {

	public SimulationTable(DataBase dataBase) {
		super(dataBase);
	}

}
