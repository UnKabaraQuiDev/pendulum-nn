import org.joml.Vector2f;
import org.junit.Test;

public class GradeTest {

	static float bell(float x, Vector2f size) {
		float L = Math.max(Math.abs(size.x), Math.abs(size.y));
		float sigma = L / 3.0f;
		return (float) Math.exp(-(x * x) / (2.0f * sigma * sigma));
	}

	static float grade(Vector2f position, Vector2f bounds) {
		return bell(position.x, bounds) * (1.0f - bell(position.y, new Vector2f(-(float) Math.PI, (float) Math.PI)));
	}

	@Test
	public void test() {
		System.err.println(grade(new Vector2f(1, (float) Math.PI), new Vector2f(-1, 1)));
	}

}
