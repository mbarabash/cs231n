package arena;


import java.util.Random;

import character.Character;
import character.Rogue;




public class Arena {
	public static Random generator = new Random();
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Character player1 = new Rogue();
		Character player2 = new Character();
		//you can use this or
		/*
		player2.name="Jeff";
		player2.health=50;
		player2.strength=10;
		*/
		// use this because of constructors in character class
		player2=new Character(10,60,70);
		 
        // Put a 2 after the name if characters are name the same
        if (player2.name.equals(player1.name)){
            player2.name += " 2";
        }
		
		
		System.out.println(player1.name+" vs."+player2.name);
		System.out.println(player1.health+"HP vs"+ player2.health+"HP");
        int turns = 0;
        
        
        
        while(player1.isAlive() && player2.isAlive()){
            turns++;
             
            System.out.println("Turn " + turns + "\n" + player1.name + ": " + player1.health + " Health | " + player2.name + ": " + player2.health + " Health\n");
            int damage;
            // player 1 attack
            damage = player1.attack(player2);
            System.out.println(player1.name + " hits " + player2.name + " for " + damage + " damage.");
             
            // player 2 attack
            damage = player2.attack(player1);
            System.out.println(player2.name + " hits " + player1.name + " for " + damage + " damage.\n");
        }
         
        // Check to see who won
        if( player1.isAlive()){
            System.out.println(player1.name + " wins!");
        }
        else if (player2.isAlive()){
            System.out.println(player2.name + " wins!");
        }
        else{
            System.out.println("It's a draw!");
        }
		
		
	}

}
