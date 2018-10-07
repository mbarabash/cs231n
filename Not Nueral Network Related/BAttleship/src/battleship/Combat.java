package battleship;
import java.util.InputMismatchException;
import java.util.Scanner;
import ships.Ships;
import ships.DestroyerShip;

import maps.Map;
public class Combat {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("Welcome to Battleship.\n Player 1 please select the positions of your ships.");
		Map playerOneMap = new Map ();
		Map playerTwoMap = new Map();
		Scanner inputPlayer = new Scanner (System.in);
		//setup phase
		
		int shipsPlaced=0;
		while(shipsPlaced!=1){
			System.out.println("Please type in the column for the origin that will be used by your destroyer.(In capitols)");
			try{
				colNum=inputPlayer.next().charAt(0);
				if(colNum=1){
				shipsPlaced++;
				}
				else{
					break;
				}
			}
			catch(InputMismatchException e){
			System.out.println("Please type in a letter from A to L");
			continue;
			}
			System.out.println("Please type in the row for the origin that will be used by your destroyer.");
			char numRow;
			while(){
			try{
				numRow=inputPlayer.next().charAt(0);
				if(numRow){
				shipsPlaced++;
				}
				else{
					break;
				}
			}
			catch(InputMismatchException e){
			
			}
			
		}
		int shipsSunk=0;
		//gameplay while loop
		while (shipsSunk!=1){
			//generates current map
			Map gameMap = new Map();
			String bombArea="0";
			System.out.println("        Ally Map");
			
			for (int row = 0; row < gameMap.mapPlayer.length; row++){
	            for(int col = 0; col < gameMap.mapPlayer[row].length; col++){
	                // Check each spot on the map and print the appropriate graphic
	                switch(gameMap.mapPlayer[row][col]){
	                case 0:
	                    // water
	                    System.out.print("~ ");
	                    break;
	                case 1:
	                    // shot
	                    System.out.print(". ");
	                    break;
	                case 2:
	                	//hit
	                	System.out.print("o");
	                case 3:
	                	//numbering system
	                	System.out.print(" ");
	                	break;
	                case 4:
	                	System.out.print("A ");
	                	break;
	                case 5:
	                	System.out.print("B ");
	                	break;
	                case 6:
	                	System.out.print("C ");
	                	break;
	                case 7:
	                	System.out.print("D ");
	                	break;
	                case 8:
	                	System.out.print("E ");
	                	break;
	                case 9:
	                	System.out.print("F ");
	                	break;
	                case 10:
	                	System.out.print("G ");
	                	break;
	                case 11:
	                	System.out.print("H ");
	                	break;
	                case 12:
	                	System.out.print("I ");
	                	break;
	                case 13:
	                	System.out.print("J ");
	                	break;
	                case 14:
	                	System.out.print("K ");
	                	break;
	                case 15:
	                	System.out.print("L ");
	                	break;
	                case 16:
	                	System.out.print("1 ");
	                	break;
	                case 17:
	                	System.out.print("2 ");
	                	break;
	                case 18:
	                	System.out.print("3 ");
	                	break;
	                case 19:
	                	System.out.print("4 ");
	                	break;
	                case 20:
	                	System.out.print("5 ");
	                	break;
	                case 21:
	                	System.out.print("6 ");
	                	break;
	                case 22:
	                	System.out.print("7 ");
	                	break;
	                case 23:
	                	System.out.print("8 ");
	                	break;
	                case 24:
	                	System.out.print("9 ");
	                	break;
	                case 25:
	                	System.out.print("10");
	                	break;
	                case 26:
	                	System.out.print("11");
	                	break;
	                case 27:
	                	System.out.print("12");
	                	break;
	                case 42:
	                	System.out.print("[]");
	                }
	            
	            }
	            System.out.println("\n"); 
	}
				System.out.println("-------------------------\n");
				System.out.println("        Enemy Map");
			
			
			 for (int row = 0; row < gameMap.mapEnemy.length; row++){
		            for(int col = 0; col < gameMap.mapEnemy[row].length; col++){
		                // Check each spot on the map and print the appropriate graphic
		                switch(gameMap.mapEnemy[row][col]){
		                case 0:
		                    // water
		                    System.out.print("~ ");
		                    break;
		                case 1:
		                    // shot
		                    System.out.print("o ");
		                    break;
		                case 2:
		                	//hit
		                	System.out.print("X");
		                case 3:
		                	//numbering system
		                	System.out.print(" ");
		                	break;
		                case 4:
		                	System.out.print("A ");
		                	break;
		                case 5:
		                	System.out.print("B ");
		                	break;
		                case 6:
		                	System.out.print("C ");
		                	break;
		                case 7:
		                	System.out.print("D ");
		                	break;
		                case 8:
		                	System.out.print("E ");
		                	break;
		                case 9:
		                	System.out.print("F ");
		                	break;
		                case 10:
		                	System.out.print("G ");
		                	break;
		                case 11:
		                	System.out.print("H ");
		                	break;
		                case 12:
		                	System.out.print("I ");
		                	break;
		                case 13:
		                	System.out.print("J ");
		                	break;
		                case 14:
		                	System.out.print("K ");
		                	break;
		                case 15:
		                	System.out.print("L ");
		                	break;
		                case 16:
		                	System.out.print("1 ");
		                	break;
		                case 17:
		                	System.out.print("2 ");
		                	break;
		                case 18:
		                	System.out.print("3 ");
		                	break;
		                case 19:
		                	System.out.print("4 ");
		                	break;
		                case 20:
		                	System.out.print("5 ");
		                	break; 
		                case 21:
		                	System.out.print("6 ");
		                	break;
		                case 22:
		                	System.out.print("7 ");
		                	break;
		                case 23:
		                	System.out.print("8 ");
		                	break;
		                case 24:
		                	System.out.print("9 ");
		                	break;
		                case 25:
		                	System.out.print("10");
		                	break;
		                case 26:
		                	System.out.print("11");
		                	break;
		                case 27:
		                	System.out.print("12");
		                	break;
		                }
		                
		            }
		            System.out.println("\n");
			 }
	          
	    		
	    		
	            
	            
		}
	}
}
