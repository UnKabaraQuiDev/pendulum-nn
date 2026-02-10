package lu.kbra.pendulum_nn.client;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class PAClientMain {

	public static PAClientMain INSTANCE;

	public static void main(String[] args) {
		final SpringApplication app = new SpringApplication(PAClientMain.class);

		app.run(args);
	}

}
