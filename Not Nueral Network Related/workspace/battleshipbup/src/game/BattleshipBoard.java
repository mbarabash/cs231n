package game;

import java.util.Scanner;

import ships.DestroyerShip;
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
		System.out.println("We will place your destroyer ship, player " + player);
		Ship playerDestroyer = new Ship(3);
		switch(playerDestroyer.getDirection()){
		case 'U':
			
		case 'R':
		case 'D':
		case 'L':
		}
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
			System.out.println("Hit at Col: " + (char)(col + 65) + " and Row: " + row);
			map[player-1][1][row][col] = 'O';
			int playerToCheck = 0;
			if(player == 1) playerToCheck = 1;
			else playerToCheck = 0;
			map[playerToCheck][0][row][col] = '~';

		} else {
			System.out.println("Miss at Col: " + (char)(col + 65) + " and Row: " + row);
			map[player-1][1][row][col] = 'X';
		}
	}

	public int colSelector(){
		Scanner scanner = new Scanner(System.in);
		System.out.println("Please enter the col letter (in CAPS).");
		int colNum = scanner.next().charAt(0) - 65;
		return colNum;
	}

	public int rowSelector(){
		Scanner scanner = new Scanner(System.in);
		System.out.println("Please enter the row.");
		int rowNum = scanner.nextInt();
		return rowNum;
	}

	public boolean isHit(int player, int col, int row)
	{
		int playerToCheck = 0;
		if(player == 1) playerToCheck = 1;
		else playerToCheck = 0;

		return map[playerToCheck][0][row][col] != '~';
	}
}