package lu.kbra.pendulum_nn.server.db.type;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import lu.kbra.pclib.PCUtils;
import lu.kbra.pclib.db.autobuild.column.type.mysql.ColumnType;
import lu.kbra.pclib.db.autobuild.column.type.mysql.TextTypes.JsonType;
import lu.kbra.pclib.db.table.DBException;
import lu.kbra.pendulum_nn.JacksonObject;

public class JacksonColumn extends JsonType {

	protected final ObjectMapper objectMapper;

	public JacksonColumn(ObjectMapper objectMapper) {
		this.objectMapper = objectMapper;
	}

	@Override
	public Object encode(Object value) {
		if (value instanceof JacksonObject) {
			try {
				return (String) objectMapper.writeValueAsString(value);
			} catch (JsonProcessingException e) {
				throw new DBException("Exception while encoding: " + value.getClass().getName(), e);
			}
		}

		return ColumnType.unsupported(value);
	}

	@Override
	public Object decode(Object value, Type type) {
		if (value == null)
			return null;

		if (PCUtils.isSubtype(type, JacksonObject.class)) {
			try {
				if (type instanceof Class<?> clazz) {
					return objectMapper.readValue((String) value, clazz);
				} else if (type instanceof ParameterizedType pt) {
					return objectMapper.readValue((String) value, new TypeReference<>() {
						@Override
						public Type getType() {
							return pt;
						}
					});
				}
			} catch (JsonProcessingException e) {
				throw new DBException("Exception while decoding: " + type.getTypeName(), e);
			}
		}

		return ColumnType.unsupported(type);
	}

}
