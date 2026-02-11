package lu.kbra.pendulum_nn.server.db.view;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import lu.kbra.pclib.db.annotations.view.DB_View;
import lu.kbra.pclib.db.annotations.view.OrderBy;
import lu.kbra.pclib.db.annotations.view.ViewColumn;
import lu.kbra.pclib.db.annotations.view.ViewTable;
import lu.kbra.pclib.db.annotations.view.ViewTable.Type;
import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.query.QueryBuilder;
import lu.kbra.pclib.db.view.DataBaseView;
import lu.kbra.pendulum_nn.NNStructure;
import lu.kbra.pendulum_nn.server.db.data.AgentStructureData;
import lu.kbra.pendulum_nn.server.db.data.ROAgentStructureScoreData;
import lu.kbra.pendulum_nn.server.db.table.AgentPerformanceTable;
import lu.kbra.pendulum_nn.server.db.table.AgentStructureTable;
import lu.kbra.pendulum_nn.server.db.table.GenerationTable;
import lu.kbra.pendulum_nn.server.db.table.SimulationTable;

@DB_View(
		name = "agent_structure_score",
		tables = {
				@ViewTable(typeName = AgentStructureTable.class, columns = { @ViewColumn(name = "hash", asName = "structure_hash") }),
				@ViewTable(typeName = SimulationTable.class, join = Type.INNER, on = "agent_structure.hash = structure_hash"),
				@ViewTable(typeName = GenerationTable.class, join = Type.INNER, on = "simulation.id = simulation_id"),
				@ViewTable(
						typeName = AgentPerformanceTable.class,
						columns = { @ViewColumn(func = "AVG(score)", asName = "average_score") },
						join = Type.INNER,
						on = "generation.id = generation_id"
				) },
		groupBy = { "structure_hash" },
		orderBy = { @OrderBy(column = "average_score", type = OrderBy.Type.DESC) }
)
@Component
public class AgentStructureScoreView extends DataBaseView<ROAgentStructureScoreData> {

	@Autowired
	private AgentStructureTable agentStructureTable;

	public AgentStructureScoreView(DataBase dataBase) {
		super(dataBase);
	}

	public NNStructure bestAgentStructure() {
		final ROAgentStructureScoreData bestStructure = super.query(
				QueryBuilder.<ROAgentStructureScoreData>select().orderByDesc("average_score").limit(1).firstNull());
		if (bestStructure == null) {
			return null;
		}
		return agentStructureTable.load(new AgentStructureData(bestStructure.getStructureHash())).getStructure();
	}

	public List<NNStructure> bestAgentStructures(int lim) {
		final List<ROAgentStructureScoreData> bestStructures = super.query(
				QueryBuilder.<ROAgentStructureScoreData>select().orderByDesc("average_score").limit(lim).list());
		return bestStructures.stream()
				.map(c -> agentStructureTable.load(new AgentStructureData(c.getStructureHash())).getStructure())
				.toList();
	}

}
