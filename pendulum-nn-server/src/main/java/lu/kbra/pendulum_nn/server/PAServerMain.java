package lu.kbra.pendulum_nn.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class PAServerMain {

	public static PAServerMain INSTANCE;

	public static void main(String[] args) {
		final SpringApplication app = new SpringApplication(PAServerMain.class);

		app.run(args);
	}

}
