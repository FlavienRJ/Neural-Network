#ifndef NEURAL_H
#define NEURAL_H

//--------------------------------------
///Inclusion
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>

//--------------------------------------
/// declaration des types
class Network;
class Neurone;
class Connection;
class TrainData;
typedef std::vector<Neurone> Layer;
typedef std::vector<double> t_val;
typedef std::vector<Connection> Connections;

//--------------------------------------
/// Constantes
const double	TAUX_ENTRAINEMENT =	0.1;
const double	MOMENTUM =			0.1;
const int		NB_MESURE =			5;
const double	LAMBDA  =			1.5;
const double	ERREUR =			0.01;

//--------------------------------------
/// Class d'entrainement
class TrainData
{
public:
	TrainData(const std::string parFile);
	bool isEOF(void) { return trainingDataFile_.eof();}
	void getTopologie(std::vector<unsigned> & parTopologie);
	unsigned getNextInputs(t_val & parInputVal);
	unsigned getTargetOutputs(t_val & parTargetOutputVals);
	
private:
	std::ifstream trainingDataFile_;
};

//--------------------------------------
/// classe de definition des connections
class Connection
{
public:
	Connection();
	double poids_;
	double deltaPoids_;
	
private:
	static double poidsRandom(void);
};

//--------------------------------------
/// classe de definition des neurones
class Neurone
{
public:
	Neurone(unsigned parNbOutput, unsigned parMyIndex);
	inline void setOutputValue(double parOutputValue) {outputValue_ = parOutputValue;}
	inline double getOutputValue(void) const {return outputValue_;}
	void feedForward(const Layer & parPrevLayer);
	void calcOutputGradients(double parTargetVal);
	void calcHiddenGradients(const Layer & parNextLayer);
	void updateInputsPoids(Layer & parPrevLayer);
	
private:
	static double fctTransfert(double parSum);
	static double fctTransfertDerivee(double par);
	double sumDOW(const Layer & parNextLayer) const;
	static double ETA;
	static double ALPHA;
	double outputValue_;
	Connections outputPoids_;
	unsigned myIndex_;
	double gradient_;
	
};

//--------------------------------------
/// classe qui contient le reseau de neurone
class Network
{
public:
	Network(const std::vector<unsigned> & parTolopologie);
	void feedForward(const std::vector<double> & parInputValues);
	void backProp(const std::vector<double> & parTargetValues);
	void getResults(std::vector<double> & parResultValues) const;
	
	double getErreur(void) const { return error_; }
	double getErreurMoyenne(void) const { return derniereMoyenneErreur_; }
	
	t_val predict(t_val & parVal);
	
private:
	std::vector<Layer> layers_;	//layers_[nbLayer][nbNeurone]
	unsigned long nbLayers;
	double error_;
	double derniereMoyenneErreur_;
	static int nombreMesure_;
	
};
#endif