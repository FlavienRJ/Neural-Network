#ifndef utilities_hpp
#define utilities_hpp

//--------------------------------------
#include "neural.hpp"
#include "stock.hpp"

//--------------------------------------
class fChrono;
class Programme;
class ProgrammeBinaire;
class ProgrammeExemple;
class ProgrammeMultiple2;
class ProgrammeNombre;
class ProgrammeXor;

//--------------------------------------
void printVector(const std::vector<double> & parVec, const std::string parText  = "");
void requestPredict(T_val & parPredictValInput, T_val & parPredictValResult, ia::Network & parNet);
void getArgument(const int argc,const char** argv, T_val & parArg);
void training(ia::Network& parNet, ia::ReadTrainData& parTrainData, std::vector<unsigned>& parTopologie,T_val& parInputVals, T_val& parTargetVals, T_val& parResultVals);
void percentOutputNeurone(std::vector<double>& parResultValues, std::vector<double>& parPercent);

//--------------------------------------
class fChrono
{
public:
	fChrono()
	{
		srand(static_cast<unsigned>(time(NULL)));
	}
	void start(void);
	long long getDuration(void);
	friend std::ostream& operator<< (std::ostream & out,fChrono & chrono);
	
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
	void EndProgramme() const;
	
protected:
	std::unique_ptr<ia::Network> myNet_;
	
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
	
private:
	
	ia::ReadTrainData* trainingData_;
	std::vector<unsigned> topologie_;
};

//--------------------------------------
class ProgrammeExemple : public Programme
{
public:
	ProgrammeExemple(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	
private:
	ia::ReadTrainData* trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeMultiple2 : public Programme
{
public:
	ProgrammeMultiple2(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	
private:
	ia::ReadTrainData* trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeNombre : public Programme
{
public:
	ProgrammeNombre(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	
private:
	ia::ReadTrainData* trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeXor : public Programme
{
public:
	ProgrammeXor(const int argc,const char * argv[]);
	virtual void Entrainement();
	virtual void Prediction();
	
private:
	ia::ReadTrainData* trainingData_;
	std::vector<unsigned> topologie_;
	
};

//--------------------------------------
class ProgrammeFenetre
{
public:
	ProgrammeFenetre();
	void run();
	
private:
	std::unique_ptr<fWindow> fen_;
};

#endif












