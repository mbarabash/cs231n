import java.util.Scanner;
public class SwitchMenu {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner userInput = new Scanner (System.in);
		System.out.println("Choose a class\n1Warrior\n2: Rogue\n3: Wizard");
                 
        int choice = userInput.nextInt();
    
        switch(choice){
        case 1:
        	System.out.println("You picked Warrior");
        	break;
        case 2:
        	System.out.println("You picked Rogue");
        	break;
        case 3:
        	System.out.println("You picked Mage");
        	break;
        default:
        	System.out.println("That wasn't an option");
        	break;
        }
	}

}
