#ifndef NEURAL_HPP
#define NEURAL_HPP
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
#include <algorithm>
#include <thread>
#include <memory>
#include <unistd.h>

//#include <boost/iostreams/stream.hpp>

//--------------------------------------
/// declaration des types
namespace ia {
	class TrainData;
	class Connection;
	class Neurone;
	class Network;
	class Cellule;
	class Systeme;
}

using Text			= std::string;
using Layer			= std::vector<ia::Neurone>;
using T_val			= std::vector<double>;
using Connections	= std::vector<ia::Connection>;
using Topologie		= std::vector<Layer>;

//--------------------------------------
/// Constantes
const double	TAUX_ENTRAINEMENT =	0.1;
const double	MOMENTUM =			0.1;
const int		NB_MESURE =			10;
const double	LAMBDA  =			1.0;
const double	ERREUR =			0.01;

//--------------------------------------
/// Class d'entrainement
namespace ia {
	class ReadTrainData
	{
	public:
		ReadTrainData(const Text parFile);
		bool isEOF(void) { return trainingDataFile_.eof();}
		void getTopologie(std::vector<unsigned> & parTopologie);
		unsigned getNextInputs(T_val & parInputVal);
		unsigned getTargetOutputs(T_val & parTargetOutputVals);
		void calcNumberTrain();
		unsigned getNumberTrain(void) const;
		
	private:
		std::ifstream trainingDataFile_;
		unsigned nbLigne_;
	};
	
	
	//--------------------------------------
	/// classe de definition des connections
	class Connection
	{
	public:
		Connection();
		Connection(double par);
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
		void getConnectionsValues(Connections & parConnections) const;
		void setConnectionsValues(Connections & parConnections);
		inline double getTauxEntrainement(void) const { return ETA;}
		inline void setTauxEntrainement(double par) { ETA = par;}
		inline double getMomentum(void) const { return ALPHA;}
		inline void setMomentum(double par) { ALPHA = par;}
		
		
	private:

		static double fctTransfert(double parSum);
		static double fctTransfertDerivee(double par);
		double sumDOW(const Layer & parNextLayer) const;
		
		double outputValue_;
		Connections outputPoids_;
		unsigned myIndex_;
		double gradient_;
		double ETA;
		double ALPHA;
		
	};
	
	//--------------------------------------
	/// classe qui contient le reseau de neurone
	class Network
	{
	public:
		Network(const unsigned parNbInput, const unsigned parNbOutput);
		Network(const std::vector<unsigned> & parTopologie, const Text & parSave = "");
		void constructNetworkFromScratch(const std::vector<unsigned> & parTopologie);
		bool constructNetworkFromFile(const Text & parSave);
		void constructNetworkRandom(const unsigned parNbInput, const unsigned parNbOutput);
		void feedForward(const T_val & parInputValues);
		void backProp(const T_val & parTargetValues);
		void getResults(T_val & parResultValues) const;
		
		double getErreur(void) const { return error_; }
		double getErreurMoyenne(void) const { return derniereMoyenneErreur_; }
		
		T_val predict(const T_val & parVal);
		void printNeuroneConnectionsPoids(void) const;
		void getNetworkTopologie(Topologie & parTopologie) const;
		void setNbMesure(int par);
		unsigned getNbMesure() const { return nombreMesure_;}
		void saveInFile();
		
	private:
		Topologie layers_;
		unsigned long nbLayers_;
		double error_;
		double derniereMoyenneErreur_;
		int nombreMesure_;
		std::ofstream outputFile_;
		std::unique_ptr<std::ifstream> inputFile_;
		
	};
	
	//--------------------------------------
	/// classe
	class Cellule
	{
	public:
		Cellule();
		
	private:
		
		
	};
	
	//--------------------------------------
	/// classe
	class Systeme
	{
	public:
		Systeme(unsigned parNbCell = 10);
		void trainAllSysteme();
		Network& getBestNetwork();
		
	private:
		unsigned nbCellule_;
		std::vector<Network> tabCell_;
		std::unique_ptr<ReadTrainData> trainingData_;
	};
	
	
}
#endif








