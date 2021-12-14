close all
clear all
clc

x=[];
y=[];


%------------------------------------------
%1,music
load('E:\PycharmProjects\eeg\data\music\EEG_music_down4.mat','train_input','train_output');
x=[x;train_input];
y=[y;train_output];
b=max(y)+1;
%-------------------------------------------------
% 2 ,picture
load('E:\PycharmProjects\eeg\data\picture\picture_ver1.mat','train_input','train_output');
x=[x;train_input];
y=[y;train_output+b];
b=max(y)+1;

%--------------------------------
%3,color
load('E:\PycharmProjects\eeg\data\color\EEG_color_down4.mat','train_input','train_output');
x=[x;train_input];
y=[y;train_output+b];
b=max(y)+1;

%-----------------------
%4, close eye relax
load('E:\PycharmProjects\eeg\data\relax\close_eye_relax.mat','train_input','train_output');
x=[x;train_input];
y=[y;train_output+b];
b=max(y)+1;
%--------------------------
%5, open eye relax
load('E:\PycharmProjects\eeg\data\relax\open_eye_relax.mat','train_input','train_output');
x=[x;train_input];
y=[y;train_output];


train_input=x;
train_output=y;

save('E:\PycharmProjects\eeg\data\EEGdata_ver2.mat','train_input','train_output');
