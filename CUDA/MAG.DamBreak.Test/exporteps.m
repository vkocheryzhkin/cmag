function [ output_args ] = exporteps( input_args )
%exporteps: export .dat to .eps

inname = strcat('dump', input_args, '.dat');
outpath = strcat('C:\\Work\\vladimir\\Poster\\images\\dump', input_args);

 hold on
 [x0 y0 z0 w0 density0 pressure0] = textread('dump0x0001.dat','%f %f %f %f %f %f','headerlines',1);
 scatter(x0,y0,5,[0.768 0.376 0.235],'filled')
 
 [x y z w density pressure] = textread(inname,'%f %f %f %f %f %f','headerlines',1);
 scatter(x,y,5,density,'filled')
 
 saveas(gcf, 'outpath', 'eps')
 clf

end

