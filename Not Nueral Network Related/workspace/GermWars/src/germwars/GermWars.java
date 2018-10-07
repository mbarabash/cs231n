
package germwars;

import processing.core.PApplet;
import processing.core.PImage;

@SuppressWarnings("serial")
public class GermWars extends PApplet 
{
	PImage background;
	PImage paddleImage;

	public Paddle player;

	public void setup() 
	{
		size(800, 600);
		background = loadImage("/GermWars/src/data/Germ Wars/background.png");
		paddleImage = loadImage("paddle.png");

		player = new Paddle(paddleImage, width/2, height * 4/5);

	}

	public void draw() 
	{     
		//player.update();
		//background(background);

		player.drawPaddle();
	}
	public void keyPressed(){
		if(key=='d'){
			player.moveRight=true;

		}
		if(key=='a'){
			player.moveLeft=true;
		}
	}
	public void keyReleased(){
		if(key=='d'){
			player.moveRight=false;
		}
		if (key=='a'){
			player.moveLeft=false;
		}
	}
	public class Paddle{
		PImage sprite;
		float xPos;
		float yPos;

		int speed = 10;
		public boolean moveRight = false;
		public boolean moveLeft=false;

		public void update(){
			if (this.moveRight){
				this.xPos+= this.speed;
			}
			if(this.moveLeft){
				this.xPos-= this.speed;
			}
		}


		public Paddle(PImage paddleSprite, float startX, float startY){
			sprite = paddleSprite;
			xPos = startX;
			yPos = startY;
		}

		public void drawPaddle(){
			image(sprite, xPos, yPos);
		}
	}
	public static void main(String _args[]){
		PApplet.main(new String[]{germwars.GermWars.class.getName()});
	}

}