package captcha;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import javax.imageio.ImageIO;

/**
 * 
 * Generate letter image to train a network
 * 
 * Inspired by https://github.com/gbaydin/OxCaptcha/
 *
 */

public class OxGenerate {
	
	final static String dataPath = "datas"; // Path to data folder
	final static String dataGenTrainPath = dataPath + "/gen/train"; // Path to data folder
	
	
	private static final String ALPHABET = "OIX";
	private static final Random RAND = new Random(1378374134);

	static AtomicInteger cpt = new AtomicInteger();
	public static void main(String[] args) throws IOException {
		cleanDir(dataGenTrainPath);

		int nByClassTrain = 333;
		for (String s : ALPHABET.split("")) {
			cleanDir(dataGenTrainPath + "/" + s);
			for (int i = 0; i < nByClassTrain; i++) {
				generateLetter(s, dataGenTrainPath);
			}
		}
	}

	private static void cleanDir(String path) {
		if (!new File(path).exists()) new File(path).mkdirs();
		else for (var f : new File(path).listFiles()) f.delete();
	}

	private static void generateLetter(String s, String dataGenPath) throws IOException {
		OxGenerate oxCaptcha = new OxGenerate(28, 28);
		// Create background
		oxCaptcha.background(Color.BLACK);
		oxCaptcha.foreground(Color.WHITE);
		oxCaptcha.textCentered(s, 0);

		int xPeriod = RAND.nextInt(16) + 2;
		int xPhase = RAND.nextInt(14) + 2;
		int yPeriod = RAND.nextInt(16) + 2;
		int yPhase = RAND.nextInt(14) + 2;

		oxCaptcha.distortionShear(xPeriod, xPhase, yPeriod, yPhase);

		oxCaptcha.save(dataGenPath + "/" + s + "/" + cpt.incrementAndGet() + ".png");
	}

	private BufferedImage _img;
	private Graphics2D _img_g;
	private int _width;
	private int _height;
	private Color _bg_color;
	private Color _fg_color;
	private char[] _chars = new char[] {};
	private int _length = 0;
	private Font _font;
	private FontRenderContext _fontRenderContext;

	private OxGenerate(int width, int height) {
		_img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		_img_g = _img.createGraphics();
		_font = new Font("Arial", Font.PLAIN, 40);
		_img_g.setFont(_font);
		_fontRenderContext = _img_g.getFontRenderContext();
		_bg_color = Color.WHITE;
		_fg_color = Color.BLACK;

		RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		hints.add(new RenderingHints(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY));
		_img_g.setRenderingHints(hints);

		_width = width;
		_height = height;

	}

	private void background(Color color) {
		_bg_color = color;
		_img_g.setPaint(color);
		_img_g.fillRect(0, 0, _width, _height);
	}

	private void foreground(Color color) {
		_fg_color = color;
	}

	private void textCentered(String chars, int kerning) {
		int l = chars.length();
		char[] t = new char[l];
		chars.getChars(0, l, t, 0);
		int styles[] = new int[l];
		for (int i = 0; i < l; i++) {
			styles[i] = Font.PLAIN;
		}
		textCentered(t, styles, kerning);
	}

	private void textCentered(char[] chars, int[] styles, int kerning) {
		_chars = chars;
		_length = _chars.length;

		char[] cc = new char[1];
		GlyphVector gv;
		int[] gvWidths = new int[_length];
		int[] gvHeights = new int[_length];
		int width = 0;
		int height = 0;
		for (int i = 0; i < _length; i++) {
			cc[0] = _chars[i];
			_font = _font.deriveFont(styles[i]);
			_font = _font.deriveFont(24.f);
			_img_g.setFont(_font);
			_fontRenderContext = _img_g.getFontRenderContext();
			gv = _font.createGlyphVector(_fontRenderContext, cc);
			gvWidths[i] = (int) gv.getVisualBounds().getWidth();
			gvHeights[i] = (int) gv.getVisualBounds().getHeight();
			if (gvHeights[i] > height) {
				height = gvHeights[i];
			}
			width = width + gvWidths[i] + kerning + 1;
		}
		int x0 = (_width - width) / 2;
		int y0 = height + (_height - height) / 2;

		int x = x0;
		_img_g.setColor(_fg_color);
		for (int i = 0; i < _length; i++) {
			cc[0] = _chars[i];
			_font = _font.deriveFont(styles[i]);
			_img_g.setFont(_font);
			renderChar(cc, x, y0);
			x = x + gvWidths[i] + kerning + 1;
		}
		_font = _font.deriveFont(Font.PLAIN);
	}

	private void renderChar(char[] cc, int x, int y) {
		_img_g.drawChars(cc, 0, 1, x, y);
	}


	private void distortionShear(int xPeriod, int xPhase, int yPeriod, int yPhase) {
		shearX(_img_g, xPeriod, xPhase, _width, _height);
		shearY(_img_g, yPeriod, yPhase, _width, _height);
	}


	private void save(String fileName) throws IOException {
		ImageIO.write(_img, "png", new File(fileName));
	}

	private void shearX(Graphics2D g, int period, int phase, int width, int height) {
		int frames = 15;

		for (int i = 0; i < height; i++) {
			double d = (period >> 1) * Math.sin((double) i / (double) period + (6.2831853071795862D * phase) / frames);
			g.copyArea(0, i, width, 1, (int) d, 0);
			g.setColor(_bg_color);
			if (d >= 0) {
				g.drawLine(0, i, (int) d, i);
			} else {
				g.drawLine(width + (int) d, i, width, i);
			}
			g.setColor(_fg_color);

		}
	}

	private void shearY(Graphics2D g, int period, int phase, int width, int height) {
		int frames = 15;

		for (int i = 0; i < width; i++) {
			double d = (period >> 1) * Math.sin((float) i / period + (6.2831853071795862D * phase) / frames);
			g.copyArea(i, 0, 1, height, 0, (int) d);
			g.setColor(_bg_color);
			if (d >= 0) {
				g.drawLine(i, 0, i, (int) d);
			} else {
				g.drawLine(i, height + (int) d, i, height);
			}
			g.setColor(_fg_color);
		}
	}





}