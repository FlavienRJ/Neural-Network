//--------------------------------------
#include "neural.h"

//-----------TrainData------------------
//--------------------------------------
/**
 *  Constructeur de la classe d'entrainement
 */
TrainData::TrainData(const std::string parFile)
{
	trainingDataFile_.open(parFile.c_str());
}

//--------------------------------------
/**
 *  Recupere la topologie du fichier de donnee
 *  @param parTopologie tableau qui contient la topologie
 */
void TrainData::getTopologie(std::vector<unsigned> &parTopologie)
{
	std::string ligne;
	std::string label;
	
	getline(trainingDataFile_, ligne);
	std::stringstream ss(ligne);
	ss >> label;
	if (this->isEOF() || label.compare("topologie:") != 0)
		abort();
	
	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		parTopologie.push_back(n);
	}
}

//--------------------------------------
/**
 *  lis les inputs dans le fichier de donnee
 *  @param parInputVal tableau de donnee d'entree
 *  @return le nombre d'entree
 */
unsigned TrainData::getNextInputs(t_val &parInputVal)
{
	
	parInputVal.clear();
	
	std::string ligne;
	std::string label;
	
	getline(trainingDataFile_, ligne);
	std::stringstream ss(ligne);
	ss >> label;
	
	if (label.compare("in:") == 0)
	{
		double val;
		while (ss >> val) { //probleme de conversion en double avec stringstream
			parInputVal.push_back(val);
		}
	}
	return static_cast<unsigned>(parInputVal.size());
}

//--------------------------------------
/**
 *  lis les targets de sortie
 *  @param parTargetOutputVals tableau de targets de sortie
 *  @return le nombre de target
 */
unsigned TrainData::getTargetOutputs(t_val &parTargetOutputVals)
{
	parTargetOutputVals.clear();
	
	std::string ligne;
	std::string label;
	
	getline(trainingDataFile_, ligne);
	std::stringstream ss(ligne);
	ss >> label;
	
	if (label.compare("out:") == 0)
	{
		double val;
		while (ss >> val) {
			parTargetOutputVals.push_back(val);
		}
	}
	return static_cast<unsigned>(parTargetOutputVals.size());
}










//-----------Connection-----------------
//--------------------------------------
/**
 *  constructeur de connection
 */
Connection::Connection()
{
	poids_ = poidsRandom();
	//std::cout << poids_ << " : " << std::endl;
}

//--------------------------------------
/**
 *  creer un poids random
 *  @return une valeur aléatoire entre 0 et 1
 */
double Connection::poidsRandom(void)
{
	return rand()/(static_cast<double>(RAND_MAX) + 1.0);
}









//-------------Neurone------------------
//--------------------------------------
/**
 *  AJOUTER UN MECANISME D'EVOLUTION
 */
double Neurone::ETA = TAUX_ENTRAINEMENT;		//taux d'entrainement :	O.15, meilleur 0.1
double Neurone::ALPHA = MOMENTUM;	//momentum :			0.5,	O.1

//--------------------------------------
/**
 *  Constructeur de neurone
 */
Neurone::Neurone(unsigned parNbOutput, unsigned parMyIndex)
{
	for (unsigned c = 0; c < parNbOutput; ++c) {
		outputPoids_.push_back(Connection());
	}
	myIndex_ = parMyIndex; //numero de neurone dans le layer
}

//--------------------------------------
/**
 *  calcul de la sortien en fonction des entrees
 *  output=f(sum(i*w))
 *  @param parPrevLayer la couche d'avant (qui constitue les entrees)
 */
void Neurone::feedForward(const Layer &parPrevLayer)
{
	double sum = 0.0;
	
	for (unsigned n = 0; n < parPrevLayer.size(); ++n) {
		sum +=	parPrevLayer[n].getOutputValue() *
		parPrevLayer[n].outputPoids_[myIndex_].poids_;
	}
	
	outputValue_ = Neurone::fctTransfert(sum);
}

//--------------------------------------
/**
 *  caul du gradient de sortie
 *  @param parTargetVal valeur de sortie attendu
 */
void Neurone::calcOutputGradients(double parTargetVal)
{
	double delta = parTargetVal - outputValue_;						//ecart entre la theorie et la sortie
	gradient_ = delta * Neurone::fctTransfertDerivee(outputValue_);	//on applique la fonction de transfert
}

//--------------------------------------
/**
 *  calcul du gradient des couches intermediaires
 *  @param parNextLayer ref vers la couche superieur
 */
void Neurone::calcHiddenGradients(const Layer &parNextLayer)
{
	//dow : derivative output weight
	double dow = sumDOW(parNextLayer);
	gradient_ = dow * Neurone::fctTransfertDerivee(outputValue_);
}

//--------------------------------------
/**
 *  mise à jour des poids
 *
 *  @param parPrevLayer ref sur la couche d'avant
 */
void Neurone::updateInputsPoids(Layer & parPrevLayer)
{
	for (unsigned n = 0; n < parPrevLayer.size(); ++n) {
		Neurone &neurone = parPrevLayer[n];
		double ancDeltaPoids = neurone.outputPoids_[myIndex_].deltaPoids_;
		double nouvDeltaPoids = (ETA * neurone.getOutputValue() * gradient_ ) + (ALPHA * ancDeltaPoids);
		neurone.outputPoids_[myIndex_].deltaPoids_ = nouvDeltaPoids;
		neurone.outputPoids_[myIndex_].poids_ += nouvDeltaPoids;
		
	}
}

//--------------------------------------
/**
 *  fonction de transfert
 *  @param parSum valeur de la somme
 *  @return valeur de sortie
 */
double Neurone::fctTransfert(double parSum)
{
	//std::cout << val << std::endl;
	return tanh(LAMBDA * parSum);
}

//--------------------------------------
/**
 *  derivee de la fonction de transfert
 */
double Neurone::fctTransfertDerivee(double par)
{
	//2sech^2(2x)~2-8x^2 en 0
	return LAMBDA - par * par;
}

//--------------------------------------
/**
 *  Somme des derivee des poids de sortie
 *  @param parNextLayer ref vers la couche supérieur
 *  @return somme des derivee de poids
 */
double Neurone::sumDOW(const Layer &parNextLayer) const
{
	double sum = 0.0;
	
	for (unsigned n = 0; n < parNextLayer.size() - 1; ++n) {
		sum += outputPoids_[n].poids_ * parNextLayer[n].gradient_;
	}
	return sum;
}

int Network::nombreMesure_ = NB_MESURE;









//-------------Network------------------
//--------------------------------------
/**
 *  Constructeur du reseau
 */
Network::Network(const std::vector<unsigned> & parTolopologie)
{
	error_ = 0.0;
	derniereMoyenneErreur_ = 0.0;
	assert(!parTolopologie.empty());
	nbLayers = parTolopologie.size();
	
	for (unsigned i = 0; i < parTolopologie.size(); ++i) {
		unsigned nbNeurone = parTolopologie[i];
		assert(nbNeurone > 0);
		layers_.push_back(Layer());
		Layer &newLayer = layers_.back();
		bool isLastLayer = (i == (parTolopologie.size() - 1));
		unsigned nbOutput = (isLastLayer)? 0 : parTolopologie[i + 1];
		
		for (unsigned j = 0; j < (nbNeurone + 1); ++j) {
			newLayer.push_back(Neurone(nbOutput, j));
		}
		
		Neurone &biasNeurone = newLayer.back();
		biasNeurone.setOutputValue(1.0);
	}
	
	//	for (auto numLayers = 0; numLayers < nbLayers; ++numLayers) {
	//		layers_.push_back(Layer());
	//		unsigned nbOutput = (numLayers == parTolopologie.size() - 1)?
	//		0:
	//		parTolopologie[numLayers + 1];
	//		//on ajoute les neurones le le "bias" neurone --> <=
	//		for (auto numNeuron = 0; numNeuron <= parTolopologie[numLayers]; ++numNeuron) {
	//			layers_.back().push_back(Neurone(nbOutput,numNeuron));
	//			std::cout << "Ajout du neurone : " << numNeuron << " a la couche : " << numLayers <<  std::endl;
	//		}
	//		layers_.back().back().setOutputValue(1.0);
	//	}
}

//--------------------------------------
/**
 *  propagation vers les couches supérieur du reseau
 *  @param parInputValues ref du tableaus des entrees
 */
void Network::feedForward(const t_val & parInputValues)
{
	assert(parInputValues.size() == layers_[0].size() -1);
	//on assigne les valeurs d'éntree aux neurones d'entree
	for (unsigned i = 0; i < parInputValues.size(); ++i) {
		layers_[0][i].setOutputValue(parInputValues[i]);
	}
	//Propagation vers les couches supérieures
	for (auto layerNb = 1; layerNb < layers_.size(); ++layerNb) {
		Layer &currentLayer = layers_[layerNb];
		Layer &prevLayer = layers_[layerNb - 1];
		for (auto n = 0; n < layers_[layerNb].size() - 1; ++n) {
			currentLayer[n].feedForward(prevLayer);
		}
	}
}

//--------------------------------------
/**
 *  "back propagation", on calcul les erreurs, les gradients et on update les poids
 *  @param parTargetValues ref vers le tableau des targets
 */
void Network::backProp(const t_val &parTargetValues)
{
	//calculer l'erreur du réseau avec la RMS (Root Mean Square Error)
	Layer &outputLayer = layers_.back();
	error_ = 0.0;
	
	for (unsigned n = 0; n <  outputLayer.size() - 1; ++n) {
		double delta = static_cast<double>(parTargetValues[n] - outputLayer[n].getOutputValue());
		error_ += pow(delta , 2.0);
	}
	error_ /= static_cast<double>(outputLayer.size() - 1.0);
	error_ = sqrt(error_); //RMS
	
	//moyenne des mesures
	derniereMoyenneErreur_ =
	(derniereMoyenneErreur_ * nombreMesure_ + error_)
	/ (nombreMesure_+ 1.0);
	
	//calcul du gradient de sortie
	for (unsigned n = 0; n < outputLayer.size(); ++n) {
		outputLayer[n].calcOutputGradients(parTargetValues[n]);
	}
	
	//calcul du gradient des couches cachees
	for (unsigned layerNum = static_cast<unsigned>(layers_.size() - 2); layerNum > 0; --layerNum) {
		Layer &hiddenLayer = layers_[layerNum];
		Layer &nextLayer = layers_[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	
	//Pour toutes les couches, de la sortie à la premiere couche cachees on update les poids
	for (unsigned layerNum = static_cast<unsigned>(layers_.size() - 1); layerNum > 0; --layerNum) {
		Layer &layer =		layers_[layerNum];
		Layer &prevLayer =	layers_[layerNum - 1];
		
		for (unsigned n = 0; n < static_cast<unsigned>(layer.size() - 1); ++n) {
			layer[n].updateInputsPoids(prevLayer);
		}
	}
}

//--------------------------------------
/**
 *  récupération des résultats
 *  @param parResultValues ref tableau de resultat
 */
void Network::getResults(t_val & parResultValues) const
{
	parResultValues.clear();
	
	for (unsigned n = 0; n < layers_.back().size(); ++n) {
		parResultValues.push_back(layers_.back()[n].getOutputValue());
	}
}

//--------------------------------------
/**
 *  fonction de prédiction
 *  @param parVal ref du tableau d'entree
 *  @return le tableau de resultat
 */
t_val Network::predict(t_val & parVal)
{
	feedForward(parVal);
	t_val resultat;
	getResults(resultat);
	return resultat;
}