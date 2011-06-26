 [x y z w density pressure] = textread('dump1x6.dat','%f %f %f %f %f %f','headerlines',1);
 scatter(x,y,10,pressure)