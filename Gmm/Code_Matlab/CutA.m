[x,fs] = audioread('62.wav');
n=0;
start_time = 1.5+3*n;
end_time = 4.1+3*n;
Y_new=x((fs*start_time+1):fs*end_time,1);
audiowrite('060_missile_move.wav',Y_new,fs);