package ships;

public class Ship {
	protected int length;
	protected int originCol;
	protected int originRow;
	protected char direction; // u, r, d, l
	
	public Ship(){
		length = 0;
		direction = 'r';
		originCol = 0;
		originRow = 0;
	}
	
	public static boolean shipFits(Ship ship)
	{
		if(ship.length == 0) return false;
		if(ship.originCol < 1 || ship.originCol > 13) return false;
		if(ship.originRow < 1 || ship.originRow > 13) return false;
		
		switch(ship.direction){
		case 'u':
			return (ship.originRow >= ship.length);
		case 'r':
			return (ship.originCol <= 13-ship.length);
		case 'd':
			return (ship.originRow <= 13-ship.length);
		case 'l':
			return (ship.originCol >= ship.length);
		}
		
		return false;
	}
	
	
}