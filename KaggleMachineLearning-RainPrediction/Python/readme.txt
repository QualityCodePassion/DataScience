This is the python code I used for the following Kaggle competition

How Much Did It Rain? II: https://www.kaggle.com/c/how-much-did-it-rain-ii

-------------------------------------------------------------------------------

I finished in the top 16% for this competition and my full Kaggle profile can 
be found here: https://www.kaggle.com/timhale/competitions

Unfortunately I put most of my effort in tuning and experimentation with my 
models, so the code is not as clean as I would like it to be and in hindsight
I should have included more comments. 

-------------------------------------------------------------------------------

Competition Description

Rainfall is highly variable across space and time, making it notoriously tricky
to measure. Rain gauges can be an effective measurement tool for a specific 
location, but it is impossible to have them everywhere. In order to have 
widespread coverage, data from weather radars is used to estimate rainfall 
nationwide. Unfortunately, these predictions never exactly match the 
measurements taken using rain gauges.


Recently, in an effort to improve their rainfall predictors, the U.S. National
Weather Service upgraded their radar network to be polarimetric. These 
polarimetric radars are able to provide higher quality data than conventional
Doppler radars because they transmit radio wave pulses with both horizontal 
and vertical orientations. 

Polarimetric radar. Image courtesy NOAA

Dual pulses make it easier to infer the size and type of precipitation because 
rain drops become flatter as they increase in size, whereas ice crystals tend
to be elongated vertically.

In this competition, you are given snapshots of polarimetric radar values and 
asked to predict the hourly rain gauge total. A word of caution: many of the 
gauge values in the training dataset are implausible (gauges may get clogged, 
for example). More details are on the data page.

-------------------------------------------------------------------------------

