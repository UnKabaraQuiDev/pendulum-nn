package lu.kbra.pendulum_nn;

import java.io.IOException;
import java.net.URI;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.WebSocketHttpHeaders;

import lu.kbra.pendulum_nn.spring_opengl.SpringGE;
import lu.rescue_rush.spring.ws_ext.client.WSExtClientHandler;
import lu.rescue_rush.spring.ws_ext.client.annotations.WSPersistentConnection;
import lu.rescue_rush.spring.ws_ext.common.annotations.WSMapping;

@Component
@WSPersistentConnection(value = true)
@WSMapping(path = "/incoming")
public class WSClient extends WSExtClientHandler {

	@Autowired
	private ApplicationContext context;
	@Autowired
	private SpringGE springGe;

	@Override
	public void onConnect(WebSocketSessionData sessionData) {
		super.onConnect(sessionData);
		super.send("/ping", "Hewwo ᓚ₍ ^. ̫ .^₎");
		if (!springGe.getEngine().isRunning()) {
			try {
				springGe.init();
			} catch (IOException e) {
				SpringApplication.exit(context, () -> 911 % 255);
			}
		}
		super.send(WSConsts.GET_RUN_AGENTS);
	}

	@WSMapping(path = "/pong")
	public void pong(WebSocketSessionData wsSessionData, String msg) {
		LOGGER.info("Got pong: " + msg);
	}

	@WSMapping(path = WSConsts.SET_SIM_CONFIG)
	public void setSimConfig(WebSocketSessionData wsSessionData, SimulationConfiguration simConfig) {
		springGe.getGameLogic().setSimConfig(simConfig);
	}

	@WSMapping(path = WSConsts.GET_RUN_AGENTS)
	public void getRunAgents(WebSocketSessionData wsSessionData, RunAgentsConfig runAgentsConfig) {
		springGe.getGameLogic().setRunAgentsConfig(runAgentsConfig);
		springGe.getGameLogic().start();
	}

	@Override
	public WebSocketHttpHeaders buildHttpHeaders() {
		return new WebSocketHttpHeaders();
	}

	@Override
	public URI buildRemoteURI() {
		return URI.create("ws://localhost:4356/");
	}

}
