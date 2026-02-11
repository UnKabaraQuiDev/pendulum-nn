package lu.kbra.pendulum_nn.server.db.data;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;

@Component
public abstract class AgentInstanceTable extends DeferredDataBaseTable<AgentInstanceData> {

	public AgentInstanceTable(DataBase dataBase) {
		super(dataBase);
	}

	public AgentInstanceData byHash(int hash) {
		return super.load(new AgentInstanceData(hash));
	}

}
