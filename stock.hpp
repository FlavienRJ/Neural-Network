#ifndef STOCK_HPP
#define STOCK_HPP
//--------------------------------------
#include "neural.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

//--------------------------------------
const unsigned DEFAULT_SIZE = 500;

//--------------------------------------
class fWidget;
class fWindow;
class fChart;
class fButton;
class fLabel;

//--------------------------------------
class fWidget
{
public:
	fWidget();
	fWidget(unsigned parX, unsigned parY);
	
	
protected:
	sf::Vector2u size_;
	
private:
	
};

//--------------------------------------
class fWindow : public fWidget
{
public:
	fWindow();
	fWindow(unsigned parX, unsigned parY);
	void run();
	
private:
	void initialize();
	
	std::unique_ptr<sf::RenderWindow> myWin_;
	sf::Event e_;
	Text title_;
	
	std::unique_ptr<fButton>	button1_;
	std::unique_ptr<fChart>		chart1_;
	
	std::vector<double> testVector;
	std::vector<sf::Vector2u> testFormatedData;
	
};

//--------------------------------------
class fChart : public fWidget
{
public:
	fChart();
	void setPosition(sf::Vector2f& parPos);
	sf::Vector2f getPosition() const;
	void setSize(sf::Vector2f& parSize);
	sf::Vector2f getSize() const;
	sf::RectangleShape& renderBackGround();
	std::vector<sf::Vector2u>& renderData(std::vector<double> & parData);
	
private:
	sf::RectangleShape			rect_;
	sf::Vector2u				pos_;
	std::vector<double>			data_;
	std::vector<sf::Vector2u>	formatedData_;
	
};

//--------------------------------------
class fButton : public fWidget
{
public:
	fButton();
	void setPosition(sf::Vector2f& parPos);
	sf::Vector2f getPosition() const;
	void setSize(sf::Vector2f& parSize);
	sf::RectangleShape& render(sf::Vector2i parPosMouse);
	void pressed();
	void released();
	bool is_in_(sf::Vector2i parPos);
	
private:
	
	sf::RectangleShape rect_;
	unsigned transparency_;
};

//--------------------------------------
class fLabel : public fWidget
{
public:
	fLabel();
	
private:
	
};
#endif