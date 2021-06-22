#!/usr/bin/env python
# coding: utf-8

# Created Planet class with attributes Pname: planet name, Ptype: planet type, Pmass:  planet mass,  Pdistance: planet distance and created accessor methods using get() to access attributes, created mutator methods set() to update attributes. __str__() to read and output members of the class.

# In[1]:


class Planet:#created class Planet
    #init helps us initialize attributes name, type mass, distance of class Planet
    def __init__(self, Pname, Ptype, Pmass, Pdistance):
        #self means the instance of this class: and we use this to access attributes of the class
        
        self.Pname = Pname  #all attributes of the class as initialised with this
        self.Ptype = Ptype
        self.Pmass = Pmass
        self.Pdistance = Pdistance
        
    def getName(self):      #getfunctions() are accessor functions and is used to return information about object
        return self.Pname   #getName() will return the name 
    
    def getType(self):
        return self.Ptype    #getType() will return Type
    
    def getMass(self):
        return self.Pmass     #getMass() will return Mass

    def getDistance(self):
        return self.Pdistance   #getDistance() will return Distance
    
    def setName(self, Pname):   #mutator method to can be used to update name each time in loop
        self.Pname = Pname
    
    def setType(self, Ptype):   #setfunctions() are mutator functions and is used for updating 
        self.Ptype = Ptype      #mutator method used to update type each time in loop
        
    def setMass(self, Pmass):
        self.mass = newmass     #mutator method to update mass each time in loop
        
    def setDistance(self, Pdistance):
        self.Pdistance = Pdistance   #mutator method to update distance each time in loop
    
    def __str__(self):   #str method to print them 
         return f'Planet Name:{Planet.getName(self)}\nPlanet type:{Planet.getType(self)}\nDistace from Sun (AU):{Planet.getDistance(self)}\nMass (kg):{Planet.getMass(self)}\n'
        
        


# Class Terrestrial is a subclass of base class Planet, the sub-class inherits attributes of base class, adding to that we have two more attributes: low surface integer and high surface integer.  Planet. __init__ inherits from Planet class, def __str__ overrides the parent class.

# In[2]:


class TerrestrialPlanet(Planet):
    #Terrestrial planet is a sub-class which inherits attributes from base class Planet
    def __init__(self, Pname, Ptype, Pmass, Pdistance, Plow_temp, Phigh_temp):
        #Pname, Ptype, Pmass, Pdistance inherits from parents class
        #low temperature and high temperature 
        Planet.__init__(self, Pname,Ptype, Pmass, Pdistance)
        self.low = Plow_temp  #inititializing high and low temperatures as we print them
        self.high = Phigh_temp
        
    #__str__ method reads and outputs members of a class
    def __str__(self):   #str method to print them 
         return f'Planet Name:{Planet.getName(self)}\nPlanet type:{Planet.getType(self)}\nDistace from Sun (AU):{Planet.getDistance(self)}\nMass (kg):{Planet.getMass(self)}\nSurface Temperature (degrees C):{self.low} to {self.high}\n'
        


# Class Jovian is one more subclass of base class Planet, inherits attributes from base class Planet, temperature above cloud tops is an additional attribute which this class has.  Planet.__init__ inherits name, type, mass, distance from base class. def __str__ overrides base class __str__. 
# 
# 

# In[3]:


class JovianPlanet(Planet):
    #Jovian planet is a sub-class which inherits base class Planet, unlike Terrestrial planet this base class has temperature above the cloud tops
    def __init__(self, Pname, Ptype, Pmass, Pdistance, temp_cloud):
        #init class to initialize all attributes with respect to this subclass
        Planet.__init__(self, Pname, Ptype, Pmass, Pdistance)
        #as temperature over cloud tops is not inherited created it here
        self.tempcloud = temp_cloud
        #str method to print the Jovian planet attribute 
    def __str__(self):
         return f'Planet Name:{Planet.getName(self)}\nPlanet type:{Planet.getType(self)}\nDistace from Sun (AU):{Planet.getDistance(self)}\nMass (kg):{Planet.getMass(self)}\nTemperature above cloud tops (degrees C):{self.tempcloud}\n'


# main() : 
# 
# function AUtoMiles to convert AU to millions of miles. therefore we should convert AU using AU*93. 
# 
# used try and catch for file access
# 
# created an empty dictionary : planet_dict for us to get each planet’s objects and store it here In loop 
# 
# created attributes: count, name, types, mass, distance, lowdist, highdist, tempcloud to access each attribute: count will relate count to planet’s attribute and store it. Count increments each time a planets attribute is accessed. Name, type, distance, mass: will be accessed from planet, however, whether to access terrestrial or jovian will be based on the temperature, if lowdist and high dist is 0, it means that we will be able to access jovian planet, else it is terrestrial planet, based on this logic we access them, and instantiate object of the sub-class accordingly, the object instantiated is name, we access each planet with name and travelled through the loop and accessed it. 
# 
# Now that we have accessed all attributes, out next attempt is to sort it, created an empty dictionary sorted_dict, iterate through planet_dict, sort it using get.
# 
# Once the dictionary is sorted, print the planets information
# 
# Take AU to miles function and print the distance of each planet from sun. 
# 

# In[4]:


def main(): #main function created
    #a function to convert AU to miles 
    def AUtoMiles(AU):  #function created
        return AU * 93    #to convert to miles AU*93 is done
    
    try:#use of try and except for the code to give us an error if we are unable to find the datafile
        file = open('planetdata.txt', 'r') # open() to open planedata.txt 
        planets = file.read().splitlines()   #read each line in file 
    except:   #if we don't fine planedata.txt gives or if the syntax is wrong throws an error
        print("file not found or accessibility error")
    #empty dictionary to store the planet data 
    planet_dict = {}
    #print(planet_dict)  printed this to test dictionary was created
    #count to iterate through the loop, initialize it to 0
    count = 0
    #name to access planet name, initialize it as empty string
    name = ''
    #type is to access if it is Terrestrail or Jovian, type is also initialized empty string
    types = ''
    #mass is to access and iterate planet mass through the loop
    mass = 0.0
    #distance is to access and iterate planet distance through the loop
    distance = 0.0
    #lowdist is for terrestrials low distance looping
    lowdist = 0
    #highdist is for terrestrials high distance looping
    highdist = 0
    #temperature over clouds top to get jovian planets temperature loop
    tempcloud = 0
    
    for planet_attribute in planets: #to access each planet's attribute in planets
        if count == 0:  #count 0 indicates first attribute name
            name = planet_attribute
        elif count == 1:  #count 1 indicates second attribute type
            types = planet_attribute
        elif count == 2:   #count 2 indicates third attribute distance
            distance = float(planet_attribute)
        elif count == 3:     #count 3 indicates fourth attribute mass 
            mass = float(planet_attribute)
        elif count == 4:  #count 4 indicates low distance of terrestrial
            lowdist = int(planet_attribute)
        elif count == 5:   #count 5 inicates high distance of terrestrial
            highdist = int(planet_attribute)
        else:   #if low and high distance is 0: then attribute is out if terrestrial and enters jovian planet
            tempcloud = int(planet_attribute)
            if lowdist == 0 and highdist == 0:
                #accesses Jovian Planet: object instantiation 
                planet_name = JovianPlanet(name, types, mass, distance, tempcloud)
            else:
                planet_name = TerrestrialPlanet(name, types, mass, distance, lowdist, highdist)
                
                
            #once we have created object instatntiation name we pass it in planet_dict and get the distance
            planet_dict[planet_name] = distance
            count = -1
            name = ''
            types = ''      #each planet gets accessed and iteration in loop till the last planet reaches happens
            mass = 0.0
            distance = 0.0
            lowdist = 0
            highdist = 0
            tempcloud = 0
        count += 1
    
    #create an empty dictionary to sort the planets
    sorted_dict = {}
    #sort the dictionary based on keys 
    sorted_keys = sorted(planet_dict, key=planet_dict.get)
    #iterate using the sorted value from above and get the key and value pair and store in dictionary
    for item in sorted_keys:  #data in planet dictionary is equal to the sorted dictionary  
        sorted_dict[item] = planet_dict[item]
    
    #print planets details  #access all planets and print each planets information
    for planet in list(sorted_dict.keys()):
        print(planet)
        
    #print distance : sorted
    for planet in list(sorted_dict.keys()):
        miles = AUtoMiles(planet.getDistance())
        print(f'{planet.getName()} is {miles:0.2f} million miles from the Sun')


#calling main function
main()
               


# In[ ]:




