package captcha;

public class GuessResult {
	String guess;
	double confident;
	
	public GuessResult(String guess, double confident) {
		super();
		this.guess = guess;
		this.confident = confident;
	}
	
}