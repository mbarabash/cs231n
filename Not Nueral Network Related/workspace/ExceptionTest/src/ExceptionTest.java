import java.util.InputMismatchException;
import java.util.Scanner;
public class ExceptionTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner input = new Scanner (System.in);
		float userMoney=0;
		System.out.println("Enter the amount of money your charector has: ");
		try{
		userMoney =input.nextFloat();
		}
		catch (InputMismatchException e){
			input.nextLine();
			
			userMoney=0;
			System.out.println("That's not a number. I'm going to assume you meant to say 0.");
		}//Exception error
		System.out.println("Great, you have " + userMoney +" money now.");
		
	}

}
