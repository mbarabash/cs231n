package ships;
import java.util.Scanner;

public class DestroyerShip extends Ship {
	public DestroyerShip(int col, int row, char dir){
		length = 3;
		direction = dir;
		originCol = col;
		originRow = row;
	}
	
	public static DestroyerShip makeDestroyerShip()
	{
		Scanner scanner = new Scanner(System.in);
		DestroyerShip destroyerShip = (DestroyerShip) new Ship();
		while(!Ship.shipFits(destroyerShip)){
			char colChoice = ' ';
			int colNum = 0;
			while(colChoice >= 'A' && colChoice <= 'L'){
				System.out.println("Chose a column location for your destroyer.");
				colChoice = scanner.next().charAt(0);
				colNum = colChoice - 65;
			}
			int rowChoice = 0;
			while(rowChoice >= 1 && rowChoice <= 12)
			{
				System.out.println("Chose a row location for your destroyer.");
				rowChoice = scanner.nextInt();
			}
			char directionChoice = ' ';
			while(directionChoice != 'U' || directionChoice != 'R' || directionChoice != 'D' || directionChoice != 'L'){
				System.out.println("Chose a direction (U, R, D, L).");
				directionChoice = scanner.next().charAt(0);
			}
			destroyerShip = new DestroyerShip(colNum, rowChoice, directionChoice);

		}
		scanner.close();
		return destroyerShip;
	}
	
	public char getDirection(){
		return direction;
	}
}
