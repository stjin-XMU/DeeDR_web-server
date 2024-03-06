%use SNF method to fuse two similarity matrix into one similarity matrix
clc
clear
close all
%load similarity matrix
%load the first matrix
filename_1 = ['H:\drug_net_fused__SNF_code_data\PPMI_drug_similarity\drug_feature_20200731\dataset\drugNets/drugnet_',num2str(1),'.txt'];%filename
% drug_net_1 = load(filename_1);
similarity_matrix_1 = load(filename_1)
%load the second matrix
filename_2 = ['H:\drug_net_fused__SNF_code_data\PPMI_drug_similarity\drug_feature_20200731\dataset\drugNets/drugnet_',num2str(2),'.txt'];%filename
% drug_net_2 = load(filename_2);
similarity_matrix_2 = load(filename_2)
%use SNF fuse two matrix
%set all the parameters.
K = 20;%number of neighbors, usually (10~30)
T = 20; %Number of Iterations, usually (10~20)
W = SNF({similarity_matrix_1,similarity_matrix_2}, K, T);
%iteration
for i=3:10
    filename = ['H:\drug_net_fused__SNF_code_data\PPMI_drug_similarity\drug_feature_20200731\dataset\drugNets/drugnet_',num2str(i),'.txt'];%filename
%     drug_net = load(filename);
    similarity_matrix = load(filename);
    W_matrix = SNF({W,similarity_matrix}, K, T);
    W = W_matrix;
end
%save result into txt file
save drug_net_fused_result_SNF.txt -ascii W
size(W)




% pwd