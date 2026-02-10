package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.AgentPerformanceData;

@Component
public abstract class AgentPerformanceTable extends DeferredDataBaseTable<AgentPerformanceData> {

	public AgentPerformanceTable(DataBase dataBase) {
		super(dataBase);
	}

}
