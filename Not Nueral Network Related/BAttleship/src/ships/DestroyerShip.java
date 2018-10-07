package ships;

import java.util.Scanner;

public class DestroyerShip extends Ships {

	public DestroyerShip(int col, int row, char dir){
		length=3;
		direction=dir;
		originCol=col;
		originRow=row;
		shipSunk=false;
		
		
	}
	
	public static DestroyerShip makeDestroyerShip(){
		Scanner scanner = new Scanner(System.in);
		DestroyerShip destroyerShip1=(DestroyerShip) new Ships();
		while(!Ships.shipFits(destroyerShip1)){
			char colChoice=' ';
			int colNum=0;
			while(colChoice>='A'&&colChoice<='L'){
				System.out.println("Choose a column location for your destroyer.");
				colChoice =scanner.next().charAt(0);
				colNum=colChoice-96;
			
			}
			int rowChoice=0;
			while(rowChoice>1&& rowChoice<=12){
				System.out.println("Choose a row location for your destroyer");
				rowChoice=scanner.nextInt();
				
			}
			char directionChoice  = ' ';
			while(directionChoice !='U' || directionChoice!='R'|| directionChoice!='L'|| directionChoice!='D'){
				System.out.println("Choose a direction (U,R,L,D");
				directionChoice= scanner.next().charAt(0);
				
			}
			destroyerShip1 =new DestroyerShip(colNum,rowChoice, directionChoice);
			
		}	
	scanner.close();
	return destroyerShip1;
			
			
		
	}
	
	
}
