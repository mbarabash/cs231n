package arrays;
import java.util.Random;
public class Arrays {
		
		public static void main(String[] args){
			
			String[] nameList = {"Jeff","Steve","Krogar"};
			 /*
				System.out.println(nameList[0]);
		        // This will print "Jeff"
		         
		        System.out.println(nameList[1]);
		        // This will print "Steve"
		         
		        System.out.println(nameList[2]);
		        // This will print "Krogar"
		         
		        nameList[1] = "Rob";
		        // This would change the string "Steve" to "Rob"
		        System.out.println(nameList[1]);
		        // This will now print "Rob"
			*/
		        Random randomGenerator = new Random();
		         
		        // Pick a random index
		        int randomIndex = randomGenerator.nextInt(nameList.length);
		         
		        // Print the element at the random index
		        System.out.println(nameList[randomIndex]);
		        
		        
		        //for loop will print every element in array
		        for(int index=0; index<nameList.length; index++){
		        	System.out.println(nameList[index]);
		        }
		        
		}
	
	
	
}
