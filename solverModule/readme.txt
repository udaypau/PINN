For unified physics intergrated CFD/FVM solver 
Hiereachy/flow diagram alowing for plug and play integrated

***********************************************************
         1.yaml files
         2.yaml_Unpacker
         3.architecture defenition
         4.geometric modelling + physics modelling(conatins also general continuity,b.cs,i.cs etc)
         5.constitutive modeller 
         6.trainer
         7.data analyzer
         8.progress report
         
    
*********************************
   CFD SOLVER
   requirments.txt
   main
        |-yaml unpacker+data analyzer ,final report are stated in the final report ONLY
        |-architecture defenition 
        |-geometry+physics+boundaries (user input - case specific)
        |-Optimization + progress report
        |-constitutive modeller
    yaml files
    eval
    semi-interactive command line tool


    
