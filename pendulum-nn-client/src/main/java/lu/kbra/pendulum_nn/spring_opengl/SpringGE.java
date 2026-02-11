package lu.kbra.pendulum_nn.spring_opengl;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import jakarta.annotation.PreDestroy;
import lu.kbra.pclib.PCUtils;
import lu.kbra.pclib.ThreadBuilder;
import lu.kbra.pclib.logger.GlobalLogger;
import lu.kbra.pendulum_nn.PALogic;
import lu.kbra.standalone.gameengine.GameEngine;

@Component
public class SpringGE {

	private final ExecutorService executor = Executors
			.newSingleThreadExecutor((r) -> ThreadBuilder.create(r).daemon(false).name("GameEngine-Main").build());

	@Autowired
	@Qualifier("main")
	private Properties props;
	@Autowired
	private PALogic paLogic;
	@Autowired
	private GameEngine engine;

//	@PostConstruct
	public void init() throws IOException {
		GlobalLogger.INIT_DEFAULT_IF_NOT_INITIALIZED = false;
		GlobalLogger.init(PCUtils.readStringSource(props.getProperty("logs.config.file")));
		GlobalLogger.info(
				"Removed " + PCUtils.deleteOldFiles(new File("./logs/"), 20) + " entries from the logs directory.");

//		engine.getWindow().getOptions().visible = false;
		
		executor.submit(engine::start); // start in other thread, will become GLFW main
	}

	@PreDestroy
	public void shutdown() throws InterruptedException {
		if (engine.isRunning()) {
			engine.stop();
		}
		Thread.sleep(1_000);
		executor.shutdownNow();
	}

	public PALogic getGameLogic() {
		return paLogic;
	}

	public GameEngine getEngine() {
		return engine;
	}

}
