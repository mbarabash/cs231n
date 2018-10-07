package booleanoperators;

import java.util.Scanner;
 
public class BooleanOperators {
 
    public static void main(String[] args) {
         
         
        Scanner userInput = new Scanner(System.in);
         
        System.out.println("Choose a class:\n1 - Warrior\n2 - Thief\n3 - Mage");
         
        int playerClass = userInput.nextInt();
        
        System.out.println("There is a guard here, what do you do?\n1 - Fight him\n2 - Steal his wallet\n3 - Throw a fireball\n4 - Talk the guard into leaving\n5 - Intimidate the guard\n6 - Bond with the guard over your shared disdain for magic");
         
        int playerChoice = userInput.nextInt();
         
        /* new code goes here */
        if (playerChoice==1&& playerClass ==1){
        	 System.out.println("You use your awesome warrior skills and punch the guard");
        }
        else if (playerChoice==2&& playerClass== 2){
        	System.out.println("You're so sneaky he doesn't notice his wallet is gone for days.");
            
        }
        else if (playerChoice==3&& playerClass==3){
        	System.out.println("You throw a very bright and shiny fireball. The guard is slightly on fire.");
            
        }
        else if ((playerClass ==2 || playerClass==3)&&playerChoice==4){
        	System.out.println("You're so charming! He is convinced to wander off for a bit.");
            
        }
        else if ((playerClass == 1 || playerClass == 3 )&&playerChoice==5){
                 System.out.println("You scare the guard off.");
            
        }
        else if (!(playerClass==3)&& playerChoice==6){
        	System.out.println("You and the guard spend the better part of the evening talking about how lame mages are.");
            
        }
        else {
            System.out.println("Your attempt fails and the guard punches you.");
        }
        System.out.println("1 - Slam the door\n2 -Pick the lock\n3 - Use a wind spell\n4 - Ninja kick.\5 - Tell the door it looks wonderful today.\6 - Threaten the door.");
        int playerChoice1 = userInput.nextInt();
        
        if (playerChoice1==1&& playerClass ==1){
        	System.out.println("You charge the door and it breaks under your weight.");
        }
        else if (playerChoice1==2&& playerClass== 2){
        	System.out.println("You pick the lock, and the door comes open.");
        }
        else if (playerChoice1==3&& playerClass== 3){
        	System.out.println("You blow the door open with your spell.");
        }
        else if ((playerClass == 1 || playerClass == 2 )&&playerClass== 2){
        	System.out.println("You hit the door just right, and it is thrown off its hinges'.");
        }
        else if ((playerClass == 2 || playerClass == 3 )&& playerClass== 2){
        	System.out.println("The door blushes and explodes from excitement.");
        }
        else if ((playerClass == 1 || playerClass == 3 )&& playerClass== 2){
        	System.out.println("Doors dont have feelings.");
        }
        else{
        	System.out.println("Your attempt fails and the door explodes.");
        }
        
        userInput.close();
    }
 
}
