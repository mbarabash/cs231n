import java.util.Scanner;
public class ForEachExample {

    public static void main(String[] args) {
    	 
        String [] inventory = { "Sword", "Yellow Potion", "Tent", "Potato Masher"};
 
        for( String inventoryItem : inventory){
 
            System.out.println(inventoryItem);
 
        }
        
        
        System.out.println("What do you want to use?");
        Scanner input = new Scanner(System.in);
        String userInput = input.nextLine();
         
        // TODO: Search the array for an item and use it
        for (String useItem : inventory){
        	if(userInput.equalsIgnoreCase(useItem)){
        		System.out.println("You used the " + useItem);
        	}
        }
    }
    
	
	
}
