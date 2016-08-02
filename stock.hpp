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
	
	std::unique_ptr<fButton> button1_;
	
};

//--------------------------------------
class fChart : public fWidget
{
public:
	fChart();
	
private:
	
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