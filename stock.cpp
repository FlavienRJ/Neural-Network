#include "stock.hpp"

//--------------------------------------

//class fWidget;
//class fChart;
//class fButton;
//class fLabel;

//--------------------------------------
fWidget::fWidget()
{
	size_.x = DEFAULT_SIZE;
	size_.y = DEFAULT_SIZE;
}

//--------------------------------------
fWidget::fWidget(unsigned parX, unsigned parY)
{
	size_.x = parX;
	size_.y = parY;
}

//--------------------------------------
fWindow::fWindow() : fWidget()
{
	initialize();
}

//--------------------------------------
fWindow::fWindow(unsigned parX, unsigned parY) : fWidget(parX,parY)
{
	initialize();
}

//--------------------------------------
void fWindow::initialize()
{
	myWin_ = std::unique_ptr<sf::RenderWindow>(new sf::RenderWindow(sf::VideoMode(size_.x,size_.y),"The Stoxx Xchange"));
	myWin_->setVerticalSyncEnabled(true);
	myWin_->setSize(size_);
	
	button1_ = std::unique_ptr<fButton>(new fButton());
	sf::Vector2f x(500,250);
	button1_->setSize(x);
	x.x=100;x.y=100;
	button1_->setPosition(x);
}

//--------------------------------------
void fWindow::run()
{
	unsigned i = 0;
	sf::Color c(255,255,255);
	while (myWin_->isOpen()) {
		myWin_->clear(c);
		i++;
		while (myWin_->pollEvent(e_)) {
			
			if (e_.type == sf::Event::Closed)
				myWin_->close();
//			if (e_.MouseButtonReleased) {
//				if (e_.mouseButton.button == sf::Mouse::Left) {
//					if (button1_->is_in_(sf::Mouse::getPosition(*myWin_))) {
//						button1_->released();
//						std::cout << i << std::endl;
//					}
//				}
//			}
			if (e_.MouseButtonPressed) {
				if (e_.mouseButton.button == sf::Mouse::Left) {
					if (button1_->is_in_(sf::Mouse::getPosition(*myWin_))) {
						button1_->pressed();
						std::cout << i << std::endl;
					}
				}
			}
			
		}
		myWin_->draw(button1_->render(sf::Mouse::getPosition(*myWin_)));
		myWin_->display();
	}
}

//--------------------------------------
fChart::fChart() : fWidget()
{
	
}

//--------------------------------------
fButton::fButton() : fWidget()
{
	transparency_=15;
	rect_.setSize(static_cast<sf::Vector2f>(size_));
	sf::Color c(0,0,0,transparency_);
	rect_.setFillColor(c);
	
}

//--------------------------------------
void fButton::setPosition(sf::Vector2f &parPos)
{
	rect_.setPosition(parPos);
}

//--------------------------------------
sf::Vector2f fButton::getPosition() const
{
	return rect_.getPosition();
}

//--------------------------------------
void fButton::setSize(sf::Vector2f &parSize)
{
	rect_.setSize(parSize);
}

//--------------------------------------
sf::RectangleShape& fButton::render(sf::Vector2i parPosMouse)
{
	//std::cout << parPosMouse.x << " : " << parPosMouse.y << std::endl;
	if (is_in_(parPosMouse)) {
		transparency_ = 150;
	}
	else
	{
		transparency_ = 75;
	}
	sf::Color c(0,0,0,transparency_);
	rect_.setFillColor(c);
	return rect_;
}

//--------------------------------------
void fButton::pressed()
{
	std::cout << "pressed" << std::endl;
}
	
//--------------------------------------
void fButton::released()
{
	
	std::cout << "released" << std::endl;
}

//--------------------------------------
bool fButton::is_in_(sf::Vector2i parPos)
{
	return ((parPos.x > rect_.getPosition().x) and (parPos.x < (rect_.getPosition().x + rect_.getSize().x)) and (parPos.y > rect_.getPosition().y) and (parPos.y < rect_.getPosition().y + rect_.getSize().y));
}


























