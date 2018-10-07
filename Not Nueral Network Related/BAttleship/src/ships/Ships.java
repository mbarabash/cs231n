package ships;

public class Ships {
	
	protected int length;
	protected char direction; // caps u, r, d, l
	protected int originCol;
	protected int originRow;
	protected boolean shipSunk;
	public Ships (){
		length = 0;
		direction= 'r';
		originCol=0;
		originRow=0;
		shipSunk=false;
	}
	public static boolean shipFits(Ships ship){
		if (ship.length ==0){ return false;}
		if (ship.originCol < 1 || ship.originCol>13){ return false;}
		if (ship.originRow < 1 || ship.originRow>13){ return false;}
		
		switch(ship.direction){
		case 'U':
			return (ship.originRow<=ship.length);
		case 'R':
			return (ship.originCol<=13-ship.length);
		case 'D':
			return (ship.originRow<=13-ship.length);
		case 'L':
			return (ship.originCol >=ship.length);
		}
		System.out.println("Please enter a valid direction");
		return false;
	}
	
	
	
}
