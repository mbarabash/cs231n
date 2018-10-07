package game;

public class BattleshipGame {
	public static void main(String[] args){
		
		BattleshipBoard board = new BattleshipBoard();
		
		//board.placePlayerShips(1);
		//board.placePlayerShips(2);
		
		while(!board.isGameOver())
		{
			board.print(1, 1);
			board.processPlayerGuess(1);
			board.print(1, 1);
			if(board.isGameOver()) break;
			board.print(2, 1);
			board.processPlayerGuess(2);
			board.print(2, 1);
		}
		
	}
}
