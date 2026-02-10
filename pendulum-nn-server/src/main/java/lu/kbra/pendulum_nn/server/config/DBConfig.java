package lu.kbra.pendulum_nn.server.config;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pclib.PCUtils;
import lu.kbra.pclib.db.SpringDataBaseEntryUtils;
import lu.kbra.pclib.db.base.DataBase;
import lu.kbra.pclib.db.connector.MySQLDataBaseConnector;
import lu.kbra.pclib.db.connector.impl.DataBaseConnector;
import lu.kbra.pclib.db.utils.DataBaseEntryUtils;
import lu.kbra.pendulum_nn.JacksonObject;
import lu.kbra.pendulum_nn.server.db.type.JacksonColumn;

@Configuration
public class DBConfig {

	@Bean
	ObjectMapper objectMapper() {
		final ObjectMapper mapper = new ObjectMapper();
		mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
		return mapper;
	}

	@Bean
	File dbConnectorFile() {
		return new File("./config/dbConnector.json");
	}

	@Bean
	MySQLDataBaseConnector connector(@Qualifier("dbConnectorFile") File config, ObjectMapper objectMapper)
			throws IOException {
		if (!config.exists()) {
			config.getParentFile().mkdirs();
			config.createNewFile();
		}
		if (Files.size(Paths.get(config.getPath())) == 0) {
			objectMapper.writeValue(config,
					new MySQLDataBaseConnector("user", "pass", "localhost", MySQLDataBaseConnector.DEFAULT_PORT));
		}
		return objectMapper.readValue(config, MySQLDataBaseConnector.class);
	}

	@Bean
	JacksonColumn jacksonColumn(ObjectMapper objMapper) {
		return new JacksonColumn(objMapper);
	}

	@Bean
	DataBaseEntryUtils dataBaseEntryUtils(JacksonColumn jacksonColumn) {
		final SpringDataBaseEntryUtils dbUtils = new SpringDataBaseEntryUtils();
		dbUtils.registerClassType(c -> JacksonObject.class.isAssignableFrom(c), c -> jacksonColumn);
		return dbUtils;
	}

	@Bean
	DataBase dataBase(DataBaseConnector dbConnector, @Value("${spring.application.name}") String name,
			@Qualifier("dataBaseEntryUtils") DataBaseEntryUtils dbEntryUtils) {
		return new DataBase(dbConnector, PCUtils.camelCaseToSnakeCase(name.replace('.', '_')), dbEntryUtils);
	}

}
