package character;

import arena.Arena;

public class Character {
	public String name = "Geoff";
	public	 int strength=20;
	public int health=50;
	public int defense=5;
	//constructors
	private static String[] nameList={"Geoff", "Steve","Krogar"};
	public Character(){
		this.name=nameList[Arena.generator.nextInt(nameList.length)];
	}
	public Character (int strength,int deense, int health){
		this();
		this.strength=strength;
		this.defense=defense;
		this.health=health;
		
	}
	//end constructors
	
	
// Finished takeDamage method
 
public int takeDamage(int damage){
         
        // Apply defense formula
        int damageTaken = damage - this.defense;
         
        // Subtract the final damage number from this character's health
        this.health -= damageTaken;
         
        return damageTaken;
    }
public int attack (Character target){
	int damageDealt=this.strength +Arena.generator.nextInt(5);
	return target.takeDamage(damageDealt);
}
public boolean isAlive(){
	return this.health>0;
}

public int getHealth(){
	return this.health;
	
}

}
