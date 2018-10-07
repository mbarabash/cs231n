package ships;

import java.util.Scanner;

public class Ship {
	protected int length;
	protected int originCol;
	protected int originRow;
	protected char direction; // u, r, d, l
	
	public Ship(int length){
		this.length = length;
		direction = 'R';
		originCol = 0;
		originRow = 0;
		
		Scanner scanner = new Scanner(System.in);
		while(!shipFits()){
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
			DestroyerShip = new DestroyerShip(colNum, rowChoice, directionChoice);

		}
		scanner.close();
		return DestroyerShip;
	}
	
	public boolean shipFits()
	{
		if(length == 0) return false;
		if(originCol < 1 || originCol > 12) return false;
		if(originRow < 1 || originRow > 12) return false;
		
		switch(direction){
		case 'U':
			return (originRow >= length);
		case 'R':
			return (originCol <= 12-length);
		case 'D':
			return (originRow <= 12-length);
		case 'L':
			return (originCol >= length);
		}
		
		return false;
	}
	
	public char getDirection(){
		return direction;
	}
	
	
}