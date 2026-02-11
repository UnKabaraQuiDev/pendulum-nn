package lu.kbra.pendulum_nn.server;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import lu.kbra.pendulum_nn.ActivationFunction;
import lu.kbra.pendulum_nn.GenerationReport;
import lu.kbra.pendulum_nn.NNStructure;
import lu.kbra.pendulum_nn.RunAgentsConfig;
import lu.kbra.pendulum_nn.SimulationConfiguration;
import lu.kbra.pendulum_nn.WSConsts;
import lu.kbra.pendulum_nn.server.db.data.AgentInstanceData;
import lu.kbra.pendulum_nn.server.db.data.AgentInstanceTable;
import lu.kbra.pendulum_nn.server.db.data.AgentPerformanceData;
import lu.kbra.pendulum_nn.server.db.data.AgentStructureData;
import lu.kbra.pendulum_nn.server.db.data.GenerationData;
import lu.kbra.pendulum_nn.server.db.data.SimConfigData;
import lu.kbra.pendulum_nn.server.db.data.SimulationData;
import lu.kbra.pendulum_nn.server.db.data.StartingConditionData;
import lu.kbra.pendulum_nn.server.db.table.AgentPerformanceTable;
import lu.kbra.pendulum_nn.server.db.table.AgentStructureTable;
import lu.kbra.pendulum_nn.server.db.table.GenerationTable;
import lu.kbra.pendulum_nn.server.db.table.SimConfigTable;
import lu.kbra.pendulum_nn.server.db.table.SimulationTable;
import lu.kbra.pendulum_nn.server.db.table.StartingConditionTable;
import lu.kbra.pendulum_nn.server.db.view.AgentStructureScoreView;
import lu.rescue_rush.spring.ws_ext.common.annotations.WSMapping;
import lu.rescue_rush.spring.ws_ext.common.annotations.WSResponseMapping;
import lu.rescue_rush.spring.ws_ext.server.WSExtServerHandler;

@WSMapping(path = "/incoming")
@Component
public class IncomingWS extends WSExtServerHandler {

	@Autowired
	private AgentPerformanceTable agentPerformanceTable;
	@Autowired
	private AgentStructureTable agentStructureTable;
	@Autowired
	private GenerationTable generationTable;
	@Autowired
	private SimulationTable simulationTable;
	@Autowired
	private SimConfigTable simConfigTable;
	@Autowired
	private StartingConditionTable startingConditionTable;
	@Autowired
	private AgentInstanceTable agentInstanceTable;

	@Autowired
	private AgentStructureScoreView agentStructureScoreView;

	@Override
	public void onConnect(WebSocketSessionData sessionData) {
		super.onConnect(sessionData);

		sessionData.send(WSConsts.SET_SIM_CONFIG, new SimulationConfiguration(true));
	}

	@WSMapping(path = "/ping")
	@WSResponseMapping(path = "/pong")
	public String ping(WebSocketSessionData sessionData, String message) {
		LOGGER.info("Ping received from " + sessionData.getSession().getId() + ": " + message);
		return "pong: " + message;
	}

	@WSMapping(path = WSConsts.GET_RUN_AGENTS)
	public RunAgentsConfig getRunAgents(WebSocketSessionData sessionData) {
		final NNStructure bestStruct = agentStructureScoreView.bestAgentStructure();
		if (bestStruct == null) {
			return new RunAgentsConfig(true, new NNStructure(5, new int[] { 5, 3, 2 }, 1, ActivationFunction.TANH), List.of(), 1024, 10);
		} else {
			return new RunAgentsConfig(false, bestStruct, agentPerformanceTable.bestInstances(bestStruct, 20), 1024, 10);
		}
	}

	@WSMapping(path = WSConsts.GEN_REPORT)
	public void generationReport(WebSocketSessionData sessionData, GenerationReport report) {
		long simulationId = report.getSimulationId();
		if (simulationId <= 0) {
			final int simConfigHash = simConfigTable.loadIfExistsElseInsert(new SimConfigData(report.getSimulationConfiguration()))
					.getHash();
			final int structHash = agentStructureTable.loadIfExistsElseInsert(new AgentStructureData(report.getStructure())).getHash();
			simulationId = simulationTable.insertAndReload(new SimulationData(simConfigHash, structHash)).getId();
		}

		final int startingConditionHash = startingConditionTable
				.loadIfExistsElseInsert(new StartingConditionData(report.getStartingCondition()))
				.getHash();
		final long generationId = generationTable
				.insertAndReload(new GenerationData(report.getGenerationIndex(), simulationId, startingConditionHash))
				.getId();
		report.getTopAgents().forEach(c -> {
			final int instanceHash = agentInstanceTable.loadIfExistsElseInsert(new AgentInstanceData(c.getValue())).getHash();
			agentPerformanceTable.insertAndReload(new AgentPerformanceData(instanceHash, generationId, c.getKey()));
		});

		sessionData.send(WSConsts.SET_SIM_ID, simulationId);
	}

}
