#ifndef MAIN_HPP
#define MAIN_HPP

#include "neural.hpp"

void printVector(const std::vector<double> & parVec, const std::string parText  = "");

void requestPredict(T_val & parPredictValInput, T_val & parPredictValResult, Network & parNet);

void getArgument(const int argc,const char** argv, T_val & parArg);

void training(Network& parNet, TrainData& parTrainData, std::vector<unsigned>& parTopologie,T_val& parInputVals, T_val& parTargetVals, T_val& parResultVals);

void percentOutputNeurone(std::vector<double>& parResultValues, std::vector<double>& parPercent);


class fChrono
{
public:
	fChrono()
	{
		srand(static_cast<unsigned>(time(NULL)));
	}
	void start(void);
	long long getDuration(void);
	void printDuration(void);
	
private:
	std::chrono::high_resolution_clock::time_point time_;
};

//--------------------------------------
class Programme
{
public:
	Programme(const int argc,const char * argv[]);
	virtual void Entrainement() = 0;
	virtual void Prediction() = 0;
	
protected:
	
	T_val inputVals_;
	T_val targetVals_;
	T_val resultVals_;
	T_val predictValInput_;
	T_val predictValResult_;
	T_val arg_;
};

//--------------------------------------
class ProgrammeBinaire : public Programme
{
public:
	ProgrammeBinaire(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	void EndProgrammeBinaire();
	
private:
	Network* myNet_;
	TrainData* trainingData_;
	std::vector<unsigned> topologie_;
};

//--------------------------------------
class ProgrammeExemple : public Programme
{
public:
	ProgrammeExemple(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	virtual void Fin();
	~ProgrammeExemple();
	
private:
	Network myNet_;
	TrainData trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeMultiple2 : public Programme
{
public:
	ProgrammeMultiple2(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	virtual void Fin();
	~ProgrammeMultiple2();
	
private:
	Network myNet_;
	TrainData trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeNombre : public Programme
{
public:
	ProgrammeNombre(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	virtual void Fin();
	~ProgrammeNombre();
	
private:
	Network myNet_;
	TrainData trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeXor : public Programme
{
public:
	ProgrammeXor(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	virtual void Fin();
	~ProgrammeXor();
	
private:
	Network myNet_;
	TrainData trainingData_;
	std::vector<unsigned> topologie_;
	
};

#endif