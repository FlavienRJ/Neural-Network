//--------------------------------------
///Inclusion
#include "neural.h"

//--------------------------------------
//TODO
//-commenter le code
//-AJOUT algorithme d'evolution genetique
//	->dans les genes : la topologie du réseau, les constantes eta/alpha, poids ?
//-ajout de la sauvegarde des parametres des neurones à la fin de l'entrainement
//-ajout d'une base de donnee dans laquelle on stock les resultats jugee bon pour encore plus augmenter le fichier de test
//-uniformiser le formatage des fichiers de donnee

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

//--------------------------------------
/**
 *  fonction principale
 *  @return 0 si le programme se termine correctement
 */
int main()
{
	srand(static_cast<unsigned>(time(NULL)));
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	
	TrainData trainingData("data.txt");
	std::vector<unsigned> topologie;
	trainingData.getTopologie(topologie);
	
	Network myNet(topologie);
	
	t_val inputVals;
	t_val targetVals;
	t_val resultVals;
	int numTraining = 0;
	
	//myNet.printNeuroneConnectionsPoids();
	
	while (!trainingData.isEOF()) {
		++numTraining;
		//std::cout << "passe n° : " << numTraining << std::endl;
		
		if (trainingData.getNextInputs(inputVals) != topologie[0])
			break;
		
		//printVector(inputVals,"inputs: ");
		
		myNet.feedForward(inputVals);
		
		myNet.getResults(resultVals);
		
		resultVals.pop_back();
		//printVector(resultVals,"outputs: ");
		
		trainingData.getTargetOutputs(targetVals);
		
		//printVector(targetVals,"targets: ");
		
		assert(targetVals.size() == topologie.back());
		
		myNet.backProp(targetVals);
		
//		std::cout << "Erreur actuelle du reseau : " << std::setprecision(2) << myNet.getErreur() << std::endl;
//		std::cout << "Moyenne d'erreur : " << myNet.getErreurMoyenne() << std::endl;
		
		if ( numTraining > 100 && myNet.getErreurMoyenne() < ERREUR )
			break;
		
	}
	myNet.printNeuroneConnectionsPoids();
	//std::cout << std::endl << "Done" << std::endl;
	
	std::vector<double> predictValInput;
	predictValInput.push_back(0.3);
	predictValInput.push_back(0.5);
	std::vector<double> predictValResult;
	requestPredict(predictValInput, predictValResult, myNet);
	predictValResult.pop_back();
	printVector(predictValResult);
	
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "temps d'execution : " << duree << " ms" << std::endl;
	
    return 0;
}
