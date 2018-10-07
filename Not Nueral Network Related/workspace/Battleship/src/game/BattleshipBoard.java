package game;

import java.util.Scanner;

import ships.Ship;

public class BattleshipBoard {

	private char[][][][] map;
	private int[] rowLabels;
	private char[] colLabels;
	private final int ROWS = 12;
	private final int COLS = 12;
	private final int PLAYERS = 2;
	private final int MAPS = 2;

	public int shipsSunk=0;



	public BattleshipBoard(){
		map = new char[PLAYERS][MAPS][ROWS][COLS];
		for(int player = 0; player < PLAYERS; player++)
		{
			for(int mapNum = 0; mapNum < MAPS; mapNum++)
			{
				for(int row = 0; row < ROWS; row++){
					for(int col = 0; col < COLS; col++){
						map[player][mapNum][row][col] = '~';
					}
				}
			}

		}
		rowLabels = new int[ROWS];
		for(int row = 0; row < ROWS; row++){
			rowLabels[row] = row + 1;
		}
		colLabels = new char[ROWS];
		for(int col = 0; col < COLS; col++){
			colLabels[col] = (char) (col + 65);
		}
	}

	public void print(int player, int mapNum)
	{
		for(int row = -1; row < ROWS; row++)
		{
			if(row == -1){
				for(int col = -1; col < COLS; col++)
				{
					if(col == -1) System.out.print("   ");
					else{
						System.out.print(colLabels[col]);
						System.out.print(" ");
					}
				}
			}
			else {
				for(int col = -1; col < COLS; col++)
				{
					if(col == -1) {
						System.out.print(rowLabels[row]);
						if(rowLabels[row] <= 9) System.out.print("  ");
						else System.out.print(" ");
					}
					else {
						System.out.print(map[player-1][mapNum][row][col]);
						System.out.print(" ");
					}
				}
			}

			System.out.print('\n');
		}

	}

	public void placePlayerShips(int player){

		//AircraftCarrier

		boolean shipPlaced = true;
		do{
			shipPlaced = true;
			System.out.println("We will place your Aircraft Carrier (length of 5), player " + player);
			Ship playerAircraftCarrier = new Ship(5);
			switch(playerAircraftCarrier.getDirection()){
			case 'U':
				for(int i = playerAircraftCarrier.getOriginRow() - 1; i > playerAircraftCarrier.getOriginRow() - 1 - playerAircraftCarrier.getLength(); i--)
				{
					if(map[player-1][0][i][playerAircraftCarrier.getOriginCol()-1] != '~') shipPlaced = false;
				}
				if(shipPlaced) {
					for(int i = playerAircraftCarrier.getOriginRow() - 1; i > playerAircraftCarrier.getOriginRow() - 1 - playerAircraftCarrier.getLength(); i--)
					{
						map[player-1][0][i][playerAircraftCarrier.getOriginCol()-1] = 'A';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'R':
				for(int i = playerAircraftCarrier.getOriginCol() - 1; i < playerAircraftCarrier.getOriginCol() - 1 + playerAircraftCarrier.getLength(); i++)
				{
					if(map[player-1][0][playerAircraftCarrier.getOriginRow()-1][i] != '~') shipPlaced = false;
				}
				if(shipPlaced) {
					for(int i = playerAircraftCarrier.getOriginCol() - 1; i < playerAircraftCarrier.getOriginCol() - 1 + playerAircraftCarrier.getLength(); i++)
					{
						map[player-1][0][playerAircraftCarrier.getOriginRow()-1][i] = 'A';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'D':
				for(int i = playerAircraftCarrier.getOriginRow() - 1; i < playerAircraftCarrier.getOriginRow() - 1 + playerAircraftCarrier.getLength(); i++)
				{
					if(map[player-1][0][i][playerAircraftCarrier.getOriginCol()-1] != '~') shipPlaced = false;
				}
				if(shipPlaced) {
					for(int i = playerAircraftCarrier.getOriginRow() - 1; i < playerAircraftCarrier.getOriginRow() - 1 + playerAircraftCarrier.getLength(); i++)
					{
						map[player-1][0][i][playerAircraftCarrier.getOriginCol()-1] = 'A';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'L':
				for(int i = playerAircraftCarrier.getOriginCol() - 1; i > playerAircraftCarrier.getOriginCol() - 1 - playerAircraftCarrier.getLength(); i--)
				{
					if(map[player-1][0][playerAircraftCarrier.getOriginRow()-1][i] !='~') shipPlaced = false;
				}
				if(shipPlaced) {
					for(int i = playerAircraftCarrier.getOriginCol() - 1; i > playerAircraftCarrier.getOriginCol() - 1 - playerAircraftCarrier.getLength(); i--)
					{
						map[player-1][0][playerAircraftCarrier.getOriginRow()-1][i] = 'A';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			}
		} while(!shipPlaced);

		//battleship

		do{
			shipPlaced = true;
			System.out.println("We will place your Battleship (length of 4), player " + player);
			Ship playerBattleship = new Ship(4);
			switch(playerBattleship.getDirection()){
			case 'U':
				for(int i = playerBattleship.getOriginRow() - 1; i > playerBattleship.getOriginRow() - 1 - playerBattleship.getLength(); i--)
				{
					if(map[player-1][0][i][playerBattleship.getOriginCol()-1] != '~') shipPlaced = false;
				} if(shipPlaced) {
					for(int i = playerBattleship.getOriginRow() - 1; i > playerBattleship.getOriginRow() - 1 - playerBattleship.getLength(); i--)
					{
						map[player-1][0][i][playerBattleship.getOriginCol()-1] = 'B';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'R':
				for(int i = playerBattleship.getOriginCol() - 1; i < playerBattleship.getOriginCol() - 1 + playerBattleship.getLength(); i++)
				{
					if(map[player-1][0][playerBattleship.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerBattleship.getOriginCol() - 1; i < playerBattleship.getOriginCol() - 1 + playerBattleship.getLength(); i++)
					{
						map[player-1][0][playerBattleship.getOriginRow()-1][i] = 'B';
					} 
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'D':
				for(int i = playerBattleship.getOriginRow() - 1; i < playerBattleship.getOriginRow() - 1 + playerBattleship.getLength(); i++)
				{
					if(map[player-1][0][i][playerBattleship.getOriginCol()-1] != '~')shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerBattleship.getOriginRow() - 1; i < playerBattleship.getOriginRow() - 1 + playerBattleship.getLength(); i++)
					{
						map[player-1][0][i][playerBattleship.getOriginCol()-1] = 'B';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'L':
				for(int i = playerBattleship.getOriginCol() - 1; i > playerBattleship.getOriginCol() - 1 - playerBattleship.getLength(); i--)
				{
					if (map[player-1][0][playerBattleship.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerBattleship.getOriginCol() - 1; i > playerBattleship.getOriginCol() - 1 - playerBattleship.getLength(); i--)
					{
						map[player-1][0][playerBattleship.getOriginRow()-1][i] = 'B';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			} 
		} while(!shipPlaced);
		//Destroyer
		do{
			shipPlaced = true;
			System.out.println("We will place your Destroyer ship, player " + player);
			Ship playerDestroyer = new Ship(3);
			switch(playerDestroyer.getDirection()){
			case 'U':
				for(int i = playerDestroyer.getOriginRow() - 1; i > playerDestroyer.getOriginRow() - 1 - playerDestroyer.getLength(); i--)
				{
					if (map[player-1][0][i][playerDestroyer.getOriginCol()-1] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerDestroyer.getOriginRow() - 1; i > playerDestroyer.getOriginRow() - 1 - playerDestroyer.getLength(); i--)
					{
						map[player-1][0][i][playerDestroyer.getOriginCol()-1] = 'D';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'R':
				for(int i = playerDestroyer.getOriginCol() - 1; i < playerDestroyer.getOriginCol() - 1 + playerDestroyer.getLength(); i++)
				{
					if(map[player-1][0][playerDestroyer.getOriginRow()-1][i]!= '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerDestroyer.getOriginCol() - 1; i < playerDestroyer.getOriginCol() - 1 + playerDestroyer.getLength(); i++)
					{
						map[player-1][0][playerDestroyer.getOriginRow()-1][i] = 'D';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'D':
				for(int i = playerDestroyer.getOriginRow() - 1; i < playerDestroyer.getOriginRow() - 1 + playerDestroyer.getLength(); i++)
				{
					if (map[player-1][0][i][playerDestroyer.getOriginCol()-1] != 'D') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerDestroyer.getOriginRow() - 1; i < playerDestroyer.getOriginRow() - 1 + playerDestroyer.getLength(); i++)
					{
						map[player-1][0][i][playerDestroyer.getOriginCol()-1] = 'D';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'L':
				for(int i = playerDestroyer.getOriginCol() - 1; i > playerDestroyer.getOriginCol() - 1 - playerDestroyer.getLength(); i--)
				{
					if(map[player-1][0][playerDestroyer.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerDestroyer.getOriginCol() - 1; i > playerDestroyer.getOriginCol() - 1 - playerDestroyer.getLength(); i--)
					{
						map[player-1][0][playerDestroyer.getOriginRow()-1][i] = 'D';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			}
		} while(!shipPlaced);
		//Submarine
		do{
			shipPlaced = true;
			System.out.println("We will place your Submarine, player " + player);
			Ship playerSubmarine = new Ship(3);
			switch(playerSubmarine.getDirection()){
			case 'U':
				for(int i = playerSubmarine.getOriginRow() - 1; i > playerSubmarine.getOriginRow() - 1 - playerSubmarine.getLength(); i--)
				{
					if(map[player-1][0][i][playerSubmarine.getOriginCol()-1] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerSubmarine.getOriginRow() - 1; i > playerSubmarine.getOriginRow() - 1 - playerSubmarine.getLength(); i--)
					{
						map[player-1][0][i][playerSubmarine.getOriginCol()-1] = 'S';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'R':
				for(int i = playerSubmarine.getOriginCol() - 1; i < playerSubmarine.getOriginCol() - 1 + playerSubmarine.getLength(); i++)
				{
					if(map[player-1][0][playerSubmarine.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerSubmarine.getOriginCol() - 1; i < playerSubmarine.getOriginCol() - 1 + playerSubmarine.getLength(); i++)
					{
						map[player-1][0][playerSubmarine.getOriginRow()-1][i] = 'S';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'D':
				for(int i = playerSubmarine.getOriginRow() - 1; i < playerSubmarine.getOriginRow() - 1 + playerSubmarine.getLength(); i++)
				{
					if(map[player-1][0][i][playerSubmarine.getOriginCol()-1] != '~')shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerSubmarine.getOriginRow() - 1; i < playerSubmarine.getOriginRow() - 1 + playerSubmarine.getLength(); i++)
					{
						map[player-1][0][i][playerSubmarine.getOriginCol()-1] = 'S';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'L':
				for(int i = playerSubmarine.getOriginCol() - 1; i > playerSubmarine.getOriginCol() - 1 - playerSubmarine.getLength(); i--)
				{
					if(map[player-1][0][playerSubmarine.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerSubmarine.getOriginCol() - 1; i > playerSubmarine.getOriginCol() - 1 - playerSubmarine.getLength(); i--)
					{
						map[player-1][0][playerSubmarine.getOriginRow()-1][i] = 'S';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			}
		}while(!shipPlaced);
		//Patrol Boat
		do{
			shipPlaced = true;
			System.out.println("We will place your Patrol Boat, player " + player);
			Ship playerPatrolBoat = new Ship(2);
			switch(playerPatrolBoat.getDirection()){
			case 'U':
				for(int i = playerPatrolBoat.getOriginRow() - 1; i > playerPatrolBoat.getOriginRow() - 1 - playerPatrolBoat.getLength(); i--)
				{
					if(map[player-1][0][i][playerPatrolBoat.getOriginCol()-1] != '~') shipPlaced = false;
				} if(shipPlaced) {
					for(int i = playerPatrolBoat.getOriginRow() - 1; i > playerPatrolBoat.getOriginRow() - 1 - playerPatrolBoat.getLength(); i--)
					{
						map[player-1][0][i][playerPatrolBoat.getOriginCol()-1] = 'P';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'R':
				for(int i = playerPatrolBoat.getOriginCol() - 1; i < playerPatrolBoat.getOriginCol() - 1 + playerPatrolBoat.getLength(); i++)
				{
					if(map[player-1][0][playerPatrolBoat.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerPatrolBoat.getOriginCol() - 1; i < playerPatrolBoat.getOriginCol() - 1 + playerPatrolBoat.getLength(); i++)
					{
						map[player-1][0][playerPatrolBoat.getOriginRow()-1][i] = 'P';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'D':
				for(int i = playerPatrolBoat.getOriginRow() - 1; i < playerPatrolBoat.getOriginRow() - 1 + playerPatrolBoat.getLength(); i++)
				{
					if(map[player-1][0][i][playerPatrolBoat.getOriginCol()-1] != '~')shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerPatrolBoat.getOriginRow() - 1; i < playerPatrolBoat.getOriginRow() - 1 + playerPatrolBoat.getLength(); i++)
					{
						map[player-1][0][i][playerPatrolBoat.getOriginCol()-1] = 'P';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			case 'L':
				for(int i = playerPatrolBoat.getOriginCol() - 1; i > playerPatrolBoat.getOriginCol() - 1 - playerPatrolBoat.getLength(); i--)
				{
					if (map[player-1][0][playerPatrolBoat.getOriginRow()-1][i] != '~') shipPlaced=false;
				} if(shipPlaced) {
					for(int i = playerPatrolBoat.getOriginCol() - 1; i > playerPatrolBoat.getOriginCol() - 1 - playerPatrolBoat.getLength(); i--)
					{
						map[player-1][0][playerPatrolBoat.getOriginRow()-1][i] = 'P';
					}
				} else {
					System.out.println("That would overlap another ship.  Try again!");
				}
				break;
			}
		}while(!shipPlaced);

	}







	public boolean isGameOver(){

		boolean player1StillAlive = false;
		boolean player2StillAlive = false;

		for(int row = 0; row < ROWS; row++)
		{
			for(int col = 0; col < COLS; col++)
			{
				if(map[0][0][row][col] != '~') player1StillAlive = true;
			}
		}
		for(int row = 0; row < ROWS; row++)
		{
			for(int col = 0; col < COLS; col++)
			{
				if(map[1][0][row][col] != '~') player2StillAlive = true;
			}
		}

		if(player1StillAlive && player2StillAlive) return false;
		else {
			if(player1StillAlive) {
				System.out.println("Well done player 1!  You have sunken all the enemy ships!");
			}
			else if(player2StillAlive) {
				System.out.println("Well done player 2!  You have sunken all the enemy ships!");
			}
			return true;
		}
	}

	public void processPlayerGuess(int player){
		int col = colSelector();
		int row = rowSelector();
		if(isHit(player, col, row)) {
			System.out.println("Hit at Col: " + (char)(col + 64) + " and Row: " + row);
			map[player-1][1][row-1][col-1] = 'O';
			int playerToCheck = 0;
			if(player == 1) playerToCheck = 1;
			else playerToCheck = 0;
			map[playerToCheck][0][row-1][col-1] = '~';

		} else {
			System.out.println("Miss at Col: " + (char)(col + 64) + " and Row: " + row);
			map[player-1][1][row-1][col-1] = 'X';
		}
	}

	public int colSelector(){
		Scanner scanner = new Scanner(System.in);
		System.out.println("Please enter the collumn letter (in CAPS).");
		int colNum = scanner.next().charAt(0) - 64;
		return colNum;
	}

	public int rowSelector(){
		Scanner scanner = new Scanner(System.in);
		System.out.println("Please enter the row number.");
		int rowNum = scanner.nextInt();
		return rowNum;
	}

	public boolean isHit(int player, int col, int row)
	{
		int playerToCheck = 0;
		if(player == 1) playerToCheck = 1;
		else playerToCheck = 0;

		return map[playerToCheck][0][row-1][col-1] != '~';
	}
}