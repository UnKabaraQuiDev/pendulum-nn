package lu.kbra.pendulum_nn.config;

import java.io.IOException;
import java.io.StringReader;
import java.util.Properties;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pclib.PCUtils;
import lu.kbra.pendulum_nn.PALogic;
import lu.kbra.standalone.gameengine.GameEngine;
import lu.kbra.standalone.gameengine.graph.window.WindowOptions;

@Configuration
public class Config {

	@Bean
	ObjectMapper objectMapper() {
		final ObjectMapper mapper = new ObjectMapper();
		mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
		return mapper;
	}

//	@Bean
//	GameLogic gameLogic(PALogic paLogic) {
//		return paLogic;
//	}

	@Bean
	Properties main() throws IOException {
		final Properties props = new Properties();
		props.load(new StringReader(PCUtils.readStringSource("classpath:/config/main.properties")));
		return props;
	}

	@Bean
	GameEngine gameEngine(PALogic gameLogic, @Qualifier("main") Properties props) {
		return new GameEngine("pendulum-nn", gameLogic, new WindowOptions(props, "windowOptions"));
	}

}
