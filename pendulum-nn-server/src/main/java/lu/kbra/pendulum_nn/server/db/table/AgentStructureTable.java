package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.AgentStructureData;

@Component
public abstract class AgentStructureTable extends DeferredDataBaseTable<AgentStructureData> {

	public AgentStructureTable(DataBase dataBase) {
		super(dataBase);
	}

}
