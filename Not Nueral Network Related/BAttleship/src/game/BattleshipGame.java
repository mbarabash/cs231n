package game;

public class BattleshipGame {
	public static void main(String[] args){
		
		BattleshipBoard board = new BattleshipBoard();
		board.print(1);
		
		/*while(board.player1Destroyer.askForPlacement());
		while(board.player2Destroyer.askForPlacement());
		
		board.placePlayerShips(1);
		board.placePlayerShips(2);
		
		while(!board.isGameOver())
		{
			board.print(1);
			board.processPlayerGuess(1);
			if(board.isGameOver()) break;
			board.print(2);
			board.processPlayerGuess(1);
		}
		*/
	}
}
