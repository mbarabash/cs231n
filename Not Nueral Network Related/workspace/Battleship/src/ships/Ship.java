package ships;

import java.util.Scanner;

public class Ship {
	protected int length;
	protected int originCol;
	protected int originRow;
	protected char direction; // u, r, d, l

	public Ship(int length){
		this.length = length;

		Scanner scanner = new Scanner(System.in);
		do{
			direction = ' ';
			originCol = 0;
			originRow = 0;
			do {
				
				System.out.println("Chose a column location for your ship. (A-L in Capitols)");
				
				
				originCol = scanner.next().charAt(0) - 64;
			} while(!(originCol >= 1 && originCol <= 12));

			do {
				System.out.println("Chose a row location for your ship. (1-12)");
				originRow = scanner.nextInt();
			} while(!(originRow >= 1 && originRow <= 12));

			while(direction != 'U' && direction != 'R' && direction != 'D' && direction != 'L'){
				System.out.println("Chose a direction (U, R, D, L).");
				direction = scanner.next().charAt(0);
			}
		} while(!shipFits());
	}

	
	
	
	public boolean shipFits()
	{
		if(length == 0){
			System.out.println("ship length 0?");
			return false;
		}
		if(originCol < 1 || originCol > 12){
			System.out.println("col to small or large, try again!");
			return false;
		}
		if(originRow < 1 || originRow > 12){
			System.out.println("row to small or large, try again!");
			return false;
		}

		switch(direction){
		case 'U':
			if(originRow >= (length-1)){
				return true;
			} else {
				System.out.println("That won't fit with those parameters. Try again.");
				return false;
			}
		case 'R':
			if(originCol <= (12-length+1)){
				return true;
			} else {
				System.out.println("That won't fit with those parameters. Try again.");
				return false;
			}
		case 'D':
			if(originRow <= (12-length+1)){
				return true;
			} else {
				System.out.println("That won't fit with those parameters. Try again.");
				return false;
			}
		case 'L':
			if(originCol >= (length-1)){
				return true;
			} else {
				System.out.println("That won't fit with those parameters. Try again.");
				return false;
			}
		}

		return false;
	}

	public char getDirection(){
		return direction;
	}

	public int getOriginCol(){
		return originCol;
	}

	public int getOriginRow() {
		return originRow;
	}

	public int getLength() {
		return length;
	}

}