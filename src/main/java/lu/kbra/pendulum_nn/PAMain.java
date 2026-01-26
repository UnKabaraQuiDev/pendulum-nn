package lu.kbra.pendulum_nn;

import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.util.Properties;

import lu.kbra.standalone.gameengine.GameEngine;
import lu.kbra.standalone.gameengine.graph.window.WindowOptions;
import lu.kbra.standalone.gameengine.impl.GameLogic;
import lu.pcy113.pclib.PCUtils;
import lu.pcy113.pclib.logger.GlobalLogger;

public class PAMain {

	public static void main(String[] args) throws IOException {
		final Properties props = new Properties();
		props.load(new StringReader(PCUtils.readStringSource("classpath:/config/main.properties")));

		GlobalLogger.INIT_DEFAULT_IF_NOT_INITIALIZED = false;
		GlobalLogger.init(PCUtils.readStringSource(props.getProperty("logs.config.file")));
		GlobalLogger.info("Removed " + PCUtils.deleteOldFiles(new File("./logs/"), 20) + " entries from the logs directory.");

		final GameLogic gameLogic = new PALogic();

		final GameEngine engine = new GameEngine("pendulum-nn", gameLogic, new WindowOptions(props, "windowOptions"));
		engine.start();
	}

}
