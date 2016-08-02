//commande de compilation
//g++ -std=c++11 -framework sfml-window -framework sfml-graphics -framework sfml-system -o app main.cpp neural.cpp stock.cpp utilities.cpp
//--------------------------------------
///Inclusion
#include "stock.hpp"
#include "main.hpp"
#include "neural.hpp"
#include "utilities.hpp"

//--------------------------------------
//TODO
//-IMPLEMENTER LA FONCTION DE  CONSTRUCTION DE RESEAU NEURONAL A PARTIR DU FICHIER
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
//+creation de classe pour les différents type d'entrainement

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
	
	T_Entrainement entrainement = EXEMPLE;
	
	switch (entrainement) {
		case BINAIRE:
		{
			ProgrammeBinaire prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgramme();
			break;
		}
		case EXEMPLE:
		{
			ProgrammeExemple prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgramme();
			break;
		}
		case MULTIPLE2:
		{
			ProgrammeMultiple2 prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgramme();
			break;
		}
		case NOMBRE:
		{
			ProgrammeNombre prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgramme();
			break;
		}
		case XOR:
		{
			ProgrammeXor prog(argc,argv);
			prog.Entrainement();
			prog.Prediction();
			prog.EndProgramme();
			break;
		}
		case FENETRE:
		{
			ProgrammeFenetre prog;
			prog.run();
			break;
		}
		default:
		{
			break;
		}
	}
	std::cout << duree;
	
    return 0;
}
