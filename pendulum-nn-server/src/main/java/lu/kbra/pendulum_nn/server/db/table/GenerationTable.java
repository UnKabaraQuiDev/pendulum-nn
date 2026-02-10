package lu.kbra.pendulum_nn.server.db.table;

import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.server.db.data.GenerationData;

@Component
public abstract class GenerationTable extends DeferredDataBaseTable<GenerationData> {

	public GenerationTable(DataBase dataBase) {
		super(dataBase);
	}
	
}
