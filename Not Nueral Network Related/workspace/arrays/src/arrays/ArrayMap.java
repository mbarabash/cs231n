package arrays;
import java.util.Random;
public class ArrayMap {

	 public static void main(String[] args) {
	      Random random = new Random();
	      
	      
	      
	        // TODO: Declare a 2D array of characters to store the map
	         int [][] map= {{0,1,2,0,0,1},{0,1,2,2,1,1},{1,2,2,2,1,1},{1,1,1,1,0,1},{0,1,1,0,0,1},{0,1,2,0,1,2}};
	         
	        // TODO: Randomly generate a map
	         for (int row = 0; row < map.length; row++){
	             for(int col = 0; col < map[row].length; col++){
	            	 
	            	 map[row][col]=random.nextInt(3);
	            	 
	             }
	         }
	         
	         
	         
	         
	     // Draw the map
	        for (int row = 0; row < map.length; row++){
	            for(int col = 0; col < map[row].length; col++){
	                // Check each spot on the map and print the appropriate graphic
	                switch(map[row][col]){
	                case 0:
	                    // Water
	                    System.out.print("~ ");
	                    break;
	                case 1:
	                    // Plains
	                    System.out.print(". ");
	                    break;
	                case 2:
	                    // Mountain
	                    System.out.print("^ ");
	                    break;
	                }
	            }
	            // Line break
	            System.out.println();
	            

	        
	        }
	         
	    }
	
	
	
	
	
	
}
