package lu.kbra.pendulum_nn.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class WSExtMainServer {

	public static WSExtMainServer INSTANCE;

	public static void main(String[] args) {
		final SpringApplication app = new SpringApplication(WSExtMainServer.class);

		app.run(args);
	}

}
