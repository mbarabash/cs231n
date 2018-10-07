
import java.util.Random;
import java.util.Scanner;
import static java.lang.System.out;
public class dungeon {
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		  System.out.println("Welcome, traveler, to the Dungeon! Adventures await!");
		  int strengthStat;
		  int playerLevel=1;
		  strengthStat = 10;
		  
		  String playerName="Horhoristo";
		  Scanner playerInput = new Scanner(System.in);
		  Random randomGenerator = new Random();
		  int number =randomGenerator.nextInt(10);
		  
		  
		  
		  double playerHealth=9.0;
		  System.out.println("What is your name?");
		  playerName = playerInput.nextLine();
		  System.out.println("You are " + playerName +", the level " + playerLevel + " adventurer");
		  System.out.println("You made it to the entrance of the cave.");
		  playerLevel=playerLevel ++;
		  System.out.println("Congratulations, you are now level " + playerLevel+ ".");
		  System.out.println("You have reached a fork in the road. Choose the left or right path");
		  String playerAnswer;
		  
		  playerAnswer =playerInput.nextLine();
		  if (playerAnswer.equalsIgnoreCase("right")){
			  double rockDamage = number;
			  System.out.println("A treacherous rock ambushes you.");
			  playerHealth -=rockDamage;
			  System.out.println("You took " + rockDamage +" damage. You have "+ playerHealth + " health remaining");
			  
		  
		 }
		  else if(playerAnswer.equalsIgnoreCase("left")){
		  System.out.println("You find a strange bottle and drink it.\n You feel refreshed.");
		  playerHealth++;
		  
		  System.out.println("You find a fountain and drink from it.");
		  playerLevel++;
		  System.out.println("Blech! The water is poisonous! You took 1 damage");
		  playerHealth--;
		 }
		  else {
			  System.out.println("Error");
		 }
		  
		  
		  
	        if( playerHealth <= 0 ){ 
	 
	            System.out.println("Oh no, "
	                 + playerName +" died! They were level "
	                    + playerLevel);
	            System.exit(0);
	 
	        }
		  
		  System.out.println("Congratulations! "+ " has survived the cave of wonders!");
		  playerInput.close();
		  
		  
	}

}
