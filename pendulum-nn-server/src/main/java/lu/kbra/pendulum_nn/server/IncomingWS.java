package lu.kbra.pendulum_nn.server;

import org.springframework.stereotype.Component;

import lu.rescue_rush.spring.ws_ext.common.annotations.WSMapping;
import lu.rescue_rush.spring.ws_ext.common.annotations.WSResponseMapping;
import lu.rescue_rush.spring.ws_ext.server.WSExtServerHandler;
import lu.rescue_rush.spring.ws_ext.server.annotation.AllowAnonymous;

@AllowAnonymous
@WSMapping(path = "/incoming")
@Component
public class IncomingWS extends WSExtServerHandler {

	@Override
	public void init() {
		LOGGER.info("Test1WS (server) initialized");
		super.DEBUG = true;
	}

	@AllowAnonymous
	@WSMapping(path = "/ping")
	@WSResponseMapping(path = "/pong")
	public String ping(WebSocketSessionData sessionData, String message) {
		LOGGER.info("Ping received from " + sessionData.getSession().getId() + ": " + message);
		return "pong: " + message;
	}

}
