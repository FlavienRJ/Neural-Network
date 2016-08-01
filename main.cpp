//commande de compilation
//g++ -std=c++11 -framework sfml-window -framework sfml-graphics -framework sfml-system -o app main.cpp neural.cpp stock.cpp
//--------------------------------------
///Inclusion
#include "stock.hpp"
#include "main.hpp"

//--------------------------------------
//TODO
//-creation de classe pour les différents type d'entrainement
//-implementation d'une nouvelle classe Creature pour :
//-AJOUT algorithme d'evolution genetique
//	->dans les genes : la topologie du réseau, les constantes eta/alpha, poids ?
//-ajout d'une base de donnee dans laquelle on stock les resultats jugee bon, pour encore augmenter la taille du fichier de test
//-Lorsque peu de donnée, les répéter (3->30)
//-Convolution pour les images ?
//- travailler sur sfml pour tracer des graphiques à partir des tableau de donnée

//DONE
//+Fonction d'affichage des tableau
//+Fonction d'affichage de la matrice de connections
//+fonction d'entree de prediction
//+uniformiser le formatage des fichiers de donnee
//+ajout de la sauvegarde des parametres du reseau à la fin de l'entrainement

//--------------------------------------
void fChrono::start()
{
	time_ = std::chrono::high_resolution_clock::now();
}

//--------------------------------------
long long fChrono::getDuration()
{
	std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(t - time_).count();
}

//--------------------------------------
void fChrono::printDuration()
{
	std::cout << "temps depuis le debut : " << getDuration() << " ms" << std::endl;
}

//--------------------------------------
Programme::Programme(const int argc,const char * argv[])
{
	getArgument(argc, argv, arg_);
}
//--------------------------------------
ProgrammeBinaire::ProgrammeBinaire(const int argc,const char * argv[]) :
	Programme(argc,argv)
{
	trainingData_ = new TrainData("exemples/binaire.txt");
	trainingData_->getTopologie(topologie_);
	myNet_ = new Network(topologie_);
	myNet_->setNbMesure(trainingData_->getNumberTrain()/10);
	
}

//--------------------------------------
void ProgrammeBinaire::Entrainement()
{
	training(*myNet_,*trainingData_,topologie_,inputVals_,targetVals_,resultVals_);
}

//--------------------------------------
void ProgrammeBinaire::Prediction()
{
	if (arg_.size()) {
		predictValInput_ = arg_;
	}
	else
	{
		predictValInput_.push_back(1.0);
		predictValInput_.push_back(0.2);
	}
	
	requestPredict(predictValInput_, predictValResult_, *myNet_);
	predictValResult_.pop_back();
	printVector(predictValResult_);
	T_val percent;
	percentOutputNeurone(predictValResult_, percent);
	printVector(percent);

}

//--------------------------------------
void ProgrammeBinaire::EndProgrammeBinaire()
{
	myNet_->saveInFile();
}

//--------------------------------------
/**
 *  afficher le contenu d'un Vector
 *
 *  @param parVec  le vector à afficher
 *  @param parText le texte (optionnel)
 */
void printVector(const std::vector<double> & parVec, const std::string parText)
{
	std::cout << parText << std::endl;
	for (auto i : parVec) {
		std::cout << i << " : ";
	}
	std::cout << std::endl;
}

//--------------------------------------
void requestPredict(T_val & parPredictValInput, T_val & parPredictValResult, Network & parNet)
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

void getArgument(const int argc,const char** argv, T_val & parArg)
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
void training(Network& parNet, TrainData& parTrainData, std::vector<unsigned>& parTopologie,T_val& parInputVals, T_val& parTargetVals, T_val& parResultVals)
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
		
		//std::cout << "Erreur actuelle du reseau : " << parNet.getErreur() << std::endl;
		std::cout << "Moyenne d'erreur : " << parNet.getErreurMoyenne() << std::endl;
		
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
			parPercent[j] *= 100;
		}
	}
	else
	{
		for (int i = 0; i < parResultValues.size(); ++i) {
			sum +=parPercent[i];
		}
		for (int j = 0; j < parPercent.size(); ++j) {
			parPercent[j] /= sum;
			parPercent[j] *= 100;
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
	fChrono duree;
	duree.start();
	
	std::cout << std::setprecision(2);
	
	T_Entrainement entrainement = BINAIRE;
	
	switch (entrainement) {
		case BINAIRE:
		{
			ProgrammeBinaire prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgrammeBinaire();
			break;
		}
		case EXEMPLE:
		{
			break;
		}
		case MULTIPLE2:
		{
			break;
		}
		case NOMBRE:
		{
			break;
		}
		case XOR:
		{
			break;
		}
		default:
		{
			break;
		}
	}
	
	duree.printDuration();
	
    return 0;
}
