package lu.kbra.pendulum_nn.server.db.table;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.annotations.view.OrderBy;
import lu.kbra.pclib.db.autobuild.query.Query;
import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.table.DeferredDataBaseTable;
import lu.kbra.pendulum_nn.NNInstance;
import lu.kbra.pendulum_nn.NNStructure;
import lu.kbra.pendulum_nn.server.db.data.AgentInstanceTable;
import lu.kbra.pendulum_nn.server.db.data.AgentPerformanceData;
import lu.kbra.pendulum_nn.server.db.data.AgentInstanceData;

@Component
public abstract class AgentPerformanceTable extends DeferredDataBaseTable<AgentPerformanceData> {

	@Autowired
	private AgentInstanceTable agentInstanceTable;

	public AgentPerformanceTable(DataBase dataBase) {
		super(dataBase);
	}

	public List<NNInstance> bestInstances(NNStructure bestStruct, int lim) {
		return bestInstances(bestStruct.hashCode(), lim).stream()
				.map(AgentPerformanceData::getInstanceHash)
				.map(agentInstanceTable::byHash)
				.map(AgentInstanceData::getInstance)
				.toList();
	}

	@Query(columns = { "structure_hash" }, limit = 1, orderBy = { @OrderBy(column = "score", type = OrderBy.Type.DESC) })
	public abstract List<AgentPerformanceData> bestInstances(int structureHash, int lim);

}
