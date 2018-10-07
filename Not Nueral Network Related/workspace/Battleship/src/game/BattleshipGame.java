package game;

import java.util.Scanner;

public class BattleshipGame {
	public static void main(String[] args){
		
		Scanner scanner = new Scanner(System.in);
		BattleshipBoard board = new BattleshipBoard();
		
		board.print(1, 0);
		board.placePlayerShips(1);
		System.out.println("This is your ship map player 1");
		board.print(1, 0);
		
		System.out.println("Hit any key to continue");
		scanner.nextLine();
		
		for(int i = 0; i < 40; i++)
		{
			System.out.println();
		}
		
		board.print(2, 0);
		board.placePlayerShips(2);
		System.out.println("This is your ship map player 2");
		board.print(2, 0);
		
		System.out.println("Hit any key to continue");
		scanner.nextLine();
		for(int i = 0; i < 40; i++)
		{
			System.out.println();
		}
		System.out.println("Hit any key to continue");
		scanner.nextLine();
		
		for(int i = 0; i < 40; i++)
		{
			System.out.println();
		}
		
		while(!board.isGameOver())
		{
			System.out.println("Here are your charts player 1");
			board.print(1, 0);
			board.print(1, 1);
			board.processPlayerGuess(1);
			System.out.println("This is your guess map after the guess");
			board.print(1, 1);
			if(board.isGameOver()) break;
			
			System.out.println("Hit any key to continue");
			scanner.nextLine();
			
			for(int i = 0; i < 40; i++)
			{
				System.out.println();
			}
			
			System.out.println("Here are your charts player 2");
			board.print(2, 0);
			board.print(2, 1);
			board.processPlayerGuess(2);
			System.out.println("This is your guess map after the guess");
			board.print(2, 1);
			
			System.out.println("Hit any key to continue");
			scanner.nextLine();
			
			for(int i = 0; i < 40; i++)
			{
				System.out.println();
			}
			
		}
		
	}
}
