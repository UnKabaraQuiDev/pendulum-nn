package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.SimConfigData;

@Component
public abstract class SimConfigTable extends DeferredDataBaseTable<SimConfigData> {

	public SimConfigTable(DataBase dataBase) {
		super(dataBase);
	}

}
