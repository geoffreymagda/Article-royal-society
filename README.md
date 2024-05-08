# Article proceedings A
 
This repository contains the code related to the article "Forces in Interacting Ferromagnetic Conductors Subjected to Electrical Currents and Magnetic Fields". The code is based on the deal.II library (see https://www.dealii.org/).

It contains in particular :

* the source directory containing the source files
* the result.csv file which is where the code sends the results
*"post treatment and figures.ipynb" providing  an example of functions to plot the results in a similar fashion to the article.
* the results_4_larger contains both the numerical results of the simulations and copies of the figures in the various configurations

Important note : In the results, the values that correspond to the actual forces as described in the article are "integral_force" and "integral_force2". More precisely, it corresponds to the x component of the force over each wire.
