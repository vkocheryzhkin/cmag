interface(echo = 2);
with(plottools); 
with(plots);
plotsetup( `ps`,plotoutput=`Poiseuille.eps`,
plotoptions=`color,noborder,portrait`);

F:= 10^(-4); l:= 10^(-3); nu:= 10^(-6);

Vel:=(y,t)->F*y*(y-l)/(2*nu)+sum(4*F*l^2*sin(Pi*y*(2*n+1)/l)*exp(-(2*n+1)^2*Pi^2*nu*t/l^2)/(nu*Pi^3*(2*n+1)^3), n = 0 .. 100);

VelData0x0225 := readdata(`XVelocityYPosition0.0225.dat`, float, 2):
VelData0x045 := readdata(`XVelocityYPosition0.045.dat`, float, 2):
VelData0x1125 := readdata(`XVelocityYPosition0.1125.dat`, float, 2):
VelData0x225 := readdata(`XVelocityYPosition0.225.dat`, float, 2):
VelData1 := readdata(`XVelocityYPosition1.dat`, float, 2):

scalex := 100000:
scaley := 1000:

Vel0x0225 := scale(plot([-Vel(y, 0.0225), y, y = 0 .. l], color = green,legend ="Analytical"), scalex, scaley):
Vel0x045 := scale(plot([-Vel(y, 0.045), y, y = 0 .. l], color = green), scalex, scaley):
Vel0x1125 := scale(plot([-Vel(y, 0.1125), y, y = 0 .. l], color = green), scalex, scaley):
Vel0x225 := scale(plot([-Vel(y, 0.225), y, y = 0 .. l], color = green), scalex, scaley):
Vel1 := scale(plot([-Vel(y, 1.0), y, y = 0 .. l], color = green), scalex, scaley):

Vel0x0225Text := textplot([0.25, 0.5, "t=0.0225s"],	align = {above, right},	font = [TIMES, ROMAN, 10]):	
Vel0x045Text := textplot([0.43, 0.5, "t=0.045s"], align = {above, right},	font = [TIMES, ROMAN, 10]):
Vel0x1125Text := textplot([0.85, 0.5, "t=0.1125s"], align = {above, right},	font = [TIMES, ROMAN, 10]):
Vel0x225Text := textplot([1.12, 0.5, "t=0.225s"], align = {above, right},	font = [TIMES, ROMAN, 10]):
Vel1Text := textplot([1.28, 0.5, typeset("t=",infinity)], align = {above, right},	font = [TIMES, ROMAN, 10]):



VelDiscret0x0225 := scale(plot(VelData0x0225, style = point, symbol = cross, color = red, legend = "WCSPH"), scalex, scaley):
VelDiscret0x045 := scale(plot(VelData0x045, style = point, symbol = cross, color = red), scalex, scaley):
VelDiscret0x1125 := scale(plot(VelData0x1125, style = point, symbol = cross, color = red), scalex, scaley):
VelDiscret0x225 := scale(plot(VelData0x225, style = point, symbol = cross, color = red), scalex, scaley):
VelDiscret1 := scale(plot(VelData1, style = point, symbol = cross, color = red), scalex, scaley):



display(
	Vel0x0225,
	Vel0x0225Text,
	
	Vel0x045,
	Vel0x045Text,
	
	Vel0x1125,
	Vel0x1125Text,
	
	Vel0x225,
	Vel0x225Text,
	
	Vel1,
	Vel1Text,	
	
	VelDiscret0x0225,
	VelDiscret0x045,
	VelDiscret0x1125,
	VelDiscret0x225,
	VelDiscret1
);

