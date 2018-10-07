package character;

import arena.Arena;

public class Rogue extends Character {
	 // Rogue already has all the methods and variables of character - we only need to change what will be different.
    public int dexterity =20;
    
    
    public Rogue(){
    super();
    this.strength=8;
    this.defense=4;
    
    }
    
    public int attack (Character target){
    	boolean criticalHit=Arena.generator.nextInt(100)<dexterity;
    	int damageDealt=this.strength +Arena.generator.nextInt(5);
    	if(criticalHit){
    		damageDealt*=2;
    		System.out.println("** Critical Hit ***");
    	}
    	return target.takeDamage(damageDealt);
    	
    
    }
    
    
    
}
