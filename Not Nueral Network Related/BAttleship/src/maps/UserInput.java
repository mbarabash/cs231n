package maps;

import java.util.Scanner;

public class UserInput extends Map{

	public static int columnSelector(){
		Scanner scanner=new Scanner(System.in);
		System.out.println("Please enter the column you would like to bomb");
		char col = scanner.next().charAt(0);
		int colNum= col-96;
		return colNum;
	}
	public static int rowSelector(){
		Scanner scanner = new Scanner (System.in);
		System.out.println("Please enter the row");
		int rowNum=scanner.nextInt();
		return rowNum;
	}
}