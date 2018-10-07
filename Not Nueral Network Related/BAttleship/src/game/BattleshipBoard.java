package game;

public class BattleshipBoard {

	private char[][][] map;
	private int[] rowLabels;
	private char[] colLabels;
	private final int ROWS = 12;
	private final int COLS = 12;
	private final int PLAYERS = 2;

	public BattleshipBoard(){
		map = new char[PLAYERS][ROWS][COLS];
		for(int player = 0; player < PLAYERS; player++)
		{
			for(int row = 0; row < ROWS; row++){
				for(int col = 0; col < COLS; col++){
					map[player][row][col] = '~';
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

	public void print(int player)
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
						System.out.print(map[player-1][row][col]);
						System.out.print(" ");
					}
				}
			}

			System.out.print('\n');
		}

	}
}
