//--------------------------------------
///Inclusion
#include "neural.h"

//TODO
//

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
	
	while (!trainingData.isEOF()) {
		++numTraining;
		std::cout << "passe n° : " << numTraining << std::endl;
		
		if (trainingData.getNextInputs(inputVals) != topologie[0])
			break;
		
		std::cout << "inputs: ";
		for (unsigned i = 0; i < inputVals.size(); ++i) {
			std::cout << inputVals[i] << " ";
		}
		std::cout << std::endl;
		
		myNet.feedForward(inputVals);
		
		myNet.getResults(resultVals);
		std::cout << "outputs: ";
		for (unsigned i = 0; i < resultVals.size() - 1; ++i) {
			std::cout << resultVals[i] << " ";
		}
		std::cout << std::endl;
		
		trainingData.getTargetOutputs(targetVals);
		std::cout << "targets: ";
		for (unsigned i = 0; i < targetVals.size(); ++i) {
			std::cout << targetVals[i] << " ";
		}
		std::cout << std::endl;
		assert(targetVals.size() == topologie.back());
		
		myNet.backProp(targetVals);
		
		std::cout << "Erreur actuelle du reseau : " << std::setprecision(2) << myNet.getErreur() << std::endl;
		std::cout << "Moyenne d'erreur : " << myNet.getErreurMoyenne() << std::endl;
		
		if ( numTraining > 100 && myNet.getErreurMoyenne() < ERREUR )
			break;
		
	}
	std::cout << std::endl << "Done" << std::endl;
	
	std::vector<double> predictValInput;
	predictValInput.push_back(1.0);
	predictValInput.push_back(1.0);
	std::vector<double> predictValResult = myNet.predict(predictValInput);
	for (unsigned i = 0; i < predictValResult.size() -1; ++i) {
		std::cout << predictValResult[i];
	}
	std::cout << std::endl;
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "temps d'execution : " << duree << " ms" << std::endl;
	
    return 0;
}
