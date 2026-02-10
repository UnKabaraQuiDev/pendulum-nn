package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.StartingConditionData;

@Component
public abstract class StartingConditionTable extends DeferredDataBaseTable<StartingConditionData> {

	public StartingConditionTable(DataBase dataBase) {
		super(dataBase);
	}

}
