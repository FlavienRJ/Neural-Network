//--------------------------------------
///Inclusion
#include "neural.h"

//--------------------------------------
//TODO
//-implementation d'une nouvelle classe Creature pour :
//-AJOUT algorithme d'evolution genetique
//	->dans les genes : la topologie du réseau, les constantes eta/alpha, poids ?
//-ajout de la sauvegarde des parametres des neurones à la fin de l'entrainement
//-ajout d'une base de donnee dans laquelle on stock les resultats jugee bon, pour encore augmenter la taille du fichier de test
//-uniformiser le formatage des fichiers de donnee
//-Lorsque peu de donnée, les répéter (3->30)
//-ajouter du scripting LUA
//-Convolution pour les images ?

//DONE
//+Fonction d'affichage des tableau
//+Fonction d'affichage de la matrice de connections
//+fonction d'entree de prediction

//--------------------------------------
/**
 *  afficher le contenu d'un Vector
 *
 *  @param parVec  le vector à afficher
 *  @param parText le texte (optionnel)
 */
void printVector(const std::vector<double> & parVec, const std::string parText = "")
{
	std::cout << parText << std::endl;
	for (unsigned i = 0; i < parVec.size(); ++i) {
		std::cout << parVec[i] << " ";
	}
	std::cout << std::endl;
}

//--------------------------------------
void requestPredict(t_val & parPredictValInput, t_val & parPredictValResult, Network & parNet)
{
	double val;
	Topologie top;
	parNet.getNetworkTopologie(top);
	top[0].pop_back();
	while (parPredictValInput.empty() or (parPredictValInput.size() < top[0].size())) {
		std::cout << "Entrer une valeur : " << std::endl;
		std::cin >> val;
		if(std::isdigit(val)==0)
			parPredictValInput.push_back(val);
	}
	parPredictValResult=parNet.predict(parPredictValInput);
}

void getArgument(const int argc,const char** argv, t_val & parArg)
{
	if(!argc)
	{
		return;
	}
	double val;
	for (unsigned i = 1; i < argc; ++i)
	{
		val = strtod(argv[i], NULL);
		parArg.push_back(val);
	}
}

//--------------------------------------
void training(Network& parNet, TrainData& parTrainData, std::vector<unsigned>& parTopologie,t_val& parInputVals, t_val& parTargetVals, t_val& parResultVals)
{
	int numTraining = 0;
	while (!parTrainData.isEOF()) {
		++numTraining;
		//std::cout << "passe n° : " << numTraining << std::endl;
		
		if (parTrainData.getNextInputs(parInputVals) != parTopologie[0])
			break;
		
		//printVector(parInputVals,"inputs: ");
		
		parNet.feedForward(parInputVals);
		
		parNet.getResults(parResultVals);
		
		parResultVals.pop_back();
		//printVector(parResultVals,"outputs: ");
		
		parTrainData.getTargetOutputs(parTargetVals);
		
		//printVector(parTargetVals,"targets: ");
		
		assert(parTargetVals.size() == parTopologie.back());
		
		parNet.backProp(parTargetVals);
		
		//std::cout << "Erreur actuelle du reseau : " << std::setprecision(2) << parNet.getErreur() << std::endl;
		std::cout << "Moyenne d'erreur : " << std::setprecision(2) << parNet.getErreurMoyenne() << std::endl;
		
		if ( numTraining > 100 && parNet.getErreurMoyenne() < ERREUR )
			break;
	}

}

//--------------------------------------
void percentOutputNeurone(std::vector<double>& parResultValues, std::vector<double>& parPercent)
{
	double sum = 0.0;
	double min_val = *std::min_element(std::begin(parResultValues), std::end(parResultValues));
	parPercent = parResultValues;
	if (min_val < 0) {
		min_val = fabs(min_val);
		for (int i = 0; i < parResultValues.size(); ++i) {
			parPercent[i] = parResultValues[i] + min_val;
			sum +=parPercent[i];
		}
		for (int j = 0; j < parPercent.size(); ++j) {
			parPercent[j] /= sum;
		}
	}
	else
	{
		for (int i = 0; i < parResultValues.size(); ++i) {
			sum +=parPercent[i];
		}
		for (int j = 0; j < parPercent.size(); ++j) {
			parPercent[j] /= sum;
		}
	}
}

//--------------------------------------
/**
 *  fonction principale
 *  @return 0 si le programme se termine correctement
 */
int main(const int argc,const char * argv[])
{
	srand(static_cast<unsigned>(time(NULL)));
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	
	TrainData trainingData("data.txt");
	std::vector<unsigned> topologie;
	trainingData.getTopologie(topologie);
	
	Network myNet(topologie);
	
	unsigned nbLine = trainingData.getNumberTrain();
	myNet.setNbMesure(nbLine/10);
	
	t_val inputVals;
	t_val targetVals;
	t_val resultVals;
	
	std::vector<double> predictValInput;
	std::vector<double> predictValResult;
	t_val arg;
	getArgument(argc, argv, arg);
	
	training(myNet,trainingData,topologie,inputVals,targetVals,resultVals);
	if (arg.size()) {
		predictValInput = arg;
	}
	else
	{
		predictValInput.push_back(1.0);
		predictValInput.push_back(0.2);
	}
	requestPredict(predictValInput, predictValResult, myNet);
	predictValResult.pop_back();
	printVector(predictValResult);
	
	t_val percent;
	percentOutputNeurone(predictValResult, percent);
	printVector(percent);
	
	//myNet.printNeuroneConnectionsPoids();
	
		//myNet.printNeuroneConnectionsPoids();
	//std::cout << std::endl << "Done" << std::endl;
	
	//std::cout << std::endl << myNet.getNbMesure() << std::endl;
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "temps d'execution : " << duree << " ms" << std::endl;
	
    return 0;
}
