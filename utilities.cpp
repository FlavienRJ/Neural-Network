#include "utilities.hpp"

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
std::ostream& operator<< (std::ostream &out,fChrono & chrono)
{
	out << "temps depuis le debut : " << chrono.getDuration() << " ms" << std::endl;
	return out;
}











//--------------------------------------
Programme::Programme(const int argc,const char * argv[])
{
	getArgument(argc, argv, arg_);
}

//--------------------------------------
void Programme::EndProgramme() const
{
	myNet_->saveInFile();
}











//--------------------------------------
ProgrammeBinaire::ProgrammeBinaire(const int argc,const char * argv[]) : Programme(argc,argv)
{
	trainingData_ = new ReadTrainData("exemples/binaire.txt");
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
ProgrammeExemple::ProgrammeExemple(const int argc,const char * argv[]) : Programme(argc,argv)
{
	trainingData_ = new ReadTrainData("exemples/exemple.txt");
	trainingData_->getTopologie(topologie_);
	myNet_ = new Network(topologie_,"save.txt");
	myNet_->setNbMesure(trainingData_->getNumberTrain()/10);
	
}

//--------------------------------------
void ProgrammeExemple::Entrainement()
{
	training(*myNet_,*trainingData_,topologie_,inputVals_,targetVals_,resultVals_);
}

//--------------------------------------
void ProgrammeExemple::Prediction()
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
ProgrammeMultiple2::ProgrammeMultiple2(const int argc,const char * argv[]) : Programme(argc,argv)
{
	trainingData_ = new ReadTrainData("exemples/multiple2.txt");
	trainingData_->getTopologie(topologie_);
	myNet_ = new Network(topologie_);
	myNet_->setNbMesure(trainingData_->getNumberTrain()/10);
	
}

//--------------------------------------
void ProgrammeMultiple2::Entrainement()
{
	training(*myNet_,*trainingData_,topologie_,inputVals_,targetVals_,resultVals_);
}

//--------------------------------------
void ProgrammeMultiple2::Prediction()
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
ProgrammeNombre::ProgrammeNombre(const int argc,const char * argv[]) : Programme(argc,argv)
{
	trainingData_ = new ReadTrainData("exemples/nombre.txt");
	trainingData_->getTopologie(topologie_);
	myNet_ = new Network(topologie_);
	myNet_->setNbMesure(trainingData_->getNumberTrain()/10);
	
}

//--------------------------------------
void ProgrammeNombre::Entrainement()
{
	training(*myNet_,*trainingData_,topologie_,inputVals_,targetVals_,resultVals_);
}

//--------------------------------------
void ProgrammeNombre::Prediction()
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
ProgrammeXor::ProgrammeXor(const int argc,const char * argv[]) : Programme(argc,argv)
{
	trainingData_ = new ReadTrainData("exemples/xor.txt");
	trainingData_->getTopologie(topologie_);
	myNet_ = new Network(topologie_);
	myNet_->setNbMesure(trainingData_->getNumberTrain()/10);
	
}

//--------------------------------------
void ProgrammeXor::Entrainement()
{
	training(*myNet_,*trainingData_,topologie_,inputVals_,targetVals_,resultVals_);
}

//--------------------------------------
void ProgrammeXor::Prediction()
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
void training(Network& parNet, ReadTrainData& parTrainData, std::vector<unsigned>& parTopologie,T_val& parInputVals, T_val& parTargetVals, T_val& parResultVals)
{
	int numTraining = 0;
	while (!parTrainData.isEOF()) {
		++numTraining;
		
		if (parTrainData.getNextInputs(parInputVals) != parTopologie[0])
			break;
		
		parNet.feedForward(parInputVals);
		
		parNet.getResults(parResultVals);
		
		parResultVals.pop_back();
		
		parTrainData.getTargetOutputs(parTargetVals);
		
		
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











