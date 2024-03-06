%% Compute Jaccard Similarity Coefficient %%

Nets = {'drugDisease'};
%Nets = {'diseaseProtein'};

for i = 1 : length(Nets)
	tic
	inputID = char(strcat('../drugNets/', Nets(i), '.txt'));
	M = load(inputID);
	Sim = 1 - pdist(M, 'jaccard');
	Sim = squareform(Sim);
	Sim = Sim + eye(size(M,1));
	Sim(isnan(Sim)) = 0;
	outputID = char(strcat('../drugNets/Sim_', Nets(i), '.txt'));
	dlmwrite(outputID, Sim, '\t');
	toc
end
