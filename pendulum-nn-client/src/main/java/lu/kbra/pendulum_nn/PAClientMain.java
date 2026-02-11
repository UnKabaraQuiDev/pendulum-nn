package lu.kbra.pendulum_nn;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import lu.kbra.pendulum_nn.spring_opengl.SpringGE;

@SpringBootApplication
public class PAClientMain {

	public static PAClientMain INSTANCE;

	@Autowired
	private WSClient client;
	
	@Autowired
	private SpringGE springGE;
	
	public static void main(String[] args) {
		final SpringApplication app = new SpringApplication(PAClientMain.class);

		app.run(args);
	}

}
