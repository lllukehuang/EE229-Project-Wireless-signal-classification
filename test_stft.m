fs = 1000; 
t = 0:1/fs:2; 
y = sin(128*pi*t) + sin(256*pi*t); % sine of periods 64 and 128. 
level = 6; 
 
figure; 
windowsize = 128; 
window = hanning(windowsize); 
nfft = windowsize; 
noverlap = windowsize-1; 
[S,F,T] = spectrogram(y,window,noverlap,nfft,fs); 
imagesc(T,F,log10(abs(S))) 
set(gca,'YDir','Normal') 
xlabel('Time (secs)') 
ylabel('Freq (Hz)') 
title('Short-time Fourier Transform spectrum') 