# TEAM AISY
## HACKMSIT 2019

ADITYA SHARMA

YATHARTH KATYAL

IKSHIT BHASKAR

SIDDHANT VERMA

INTRODUCTION TO OUR WEB APP AFTERTERROR:

A Web app which detects nearest evacuation spots by evaluating the damage caused due to earthquake using a machine learning model.

FEATURES: 
• Current location of the user will be given. 
• Nearest evacuation spots would be mentioned, inorder to inform the user where they should rush immediately. 
• Affected areas would be also be shown in the WebApp. 
• Quickest navigation route towards the affected areas will also be provided. 
• A Subsection of Twitter would be given to inorder to spread quick awareness among the people, where the trending tweets will be displayed.

IMPLEMENTATION: 

We can split the Implementation section of this Application into two sub-categories – The Machine Learning Model and the Web based Application.

1)ML MODEL: 

a) We have a database of 632K + 421K Buildings. We can spread this database all over Delhi evenly through our Google Maps API. Following        this, We’ll determine the current Location of the User, using that Location, we’ll find all the buildings within a distance of 1KM around that location.

b) From the Web Server, the API will request thge current location of the USER. Using this location we need to determine all the buildings within the radius of 1KM and predict the DAMAGE GRADE of each Building from our stored DATABASE OF BUILDINGS and return it to the WEB SERVER.

2)WEB APPLICATION:

a) An online portal that stores the user’s basic information and displays live updates of relevant catastrophe occurrences around the world. We will use Angular for building the frontend of the app and Firebase as backend.

b)Intuitive and simple UI/UX designed using MaterializeCSS will be implemented. The portal tracks the user’s real time location and notifies him/her on occurrence of a nearby earthquake along with its magnitude using an open source geo location API.

c)The app displays all nearby evacuation spots or ‘Safe-Zones’ and filters them on the basis of damage caused due to the calamity using the dataset provided.

d)It then automatically finds the fastest route to the Safe-Zone and navigates the user to that location using the pre trained Machine Learning Model.

e)A sidebar with live twitter and newsroom updates on the ongoing situation will be displayed using twitter’s API and the user can tweet or push updates directly from the app in order to spread awareness and help others in need
