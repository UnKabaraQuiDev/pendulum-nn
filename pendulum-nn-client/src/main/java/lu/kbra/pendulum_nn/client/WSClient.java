package lu.kbra.pendulum_nn.client;

import java.net.URI;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.WebSocketHttpHeaders;

import lu.rescue_rush.spring.ws_ext.client.WSExtClientHandler;
import lu.rescue_rush.spring.ws_ext.client.annotations.WSPersistentConnection;
import lu.rescue_rush.spring.ws_ext.common.annotations.WSMapping;

@Component
@WSPersistentConnection(value = true)
@WSMapping(path = "/incoming")
public class WSClient extends WSExtClientHandler {

	@Override
	public WebSocketHttpHeaders buildHttpHeaders() {
		return new WebSocketHttpHeaders();
	}

	@Override
	public URI buildRemoteURI() {
		return URI.create("ws://localhost:4356/");
	}

}
