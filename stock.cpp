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
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::Image icon;
	icon.loadFromFile("logo2.png");
	
	//1500x1250
	title_ = "The Stoxx Xchange";
	myWin_ = std::unique_ptr<sf::RenderWindow>(new sf::RenderWindow(sf::VideoMode(size_.x,size_.y),title_,sf::Style::Default, settings));
	myWin_->setVerticalSyncEnabled(true);
	myWin_->setSize(size_);
	myWin_->setIcon(512, 512, icon.getPixelsPtr());
	
	
	button1_ = std::unique_ptr<fButton>(new fButton());
	sf::Vector2f x(1400,100);
	button1_->setSize(x);
	x.x=50;x.y=50;
	button1_->setPosition(x);
	
	chart1_ = std::unique_ptr<fChart>(new fChart());
	x.x=1400;x.y=1000;
	chart1_->setSize(x);
	x.x=50;x.y=200;
	chart1_->setPosition(x);
	
	
	testVector.push_back(20000);
	
	
}

//--------------------------------------
void fWindow::run()
{
	unsigned i = 0;
	unsigned j = 20000;
	unsigned w = 0;
	sf::Color c(255,255,255);
	sf::Vertex pastPoint;
	sf::Vertex currentPoint;
	sf::Vertex line[2];
	
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
		myWin_->draw(chart1_->renderBackGround());
		w = std::rand() % 6;
		//std::cout << w << std::endl;
		switch (w) {
			case 0:
				j+=25;
				break;
			case 1:
				j+=10;
				break;
			case 2:
				j+=5;
				break;
			case 3:
				j-=5;
				break;
			case 4:
				j-=10;
				break;
			case 5:
				j-=25;
				break;
			default:
				break;
		}
		testVector.push_back(j);
		testFormatedData = chart1_->renderData(testVector);
		
		title_ ="The Stoxx Xchange " + std::to_string(i) + " : " + std::to_string(j) ;
		myWin_->setTitle(title_);
		
		for (unsigned n = 0; n < testFormatedData.size() - 1; ++n) {
			line[0] = sf::Vertex(static_cast<sf::Vector2f>(testFormatedData[n]),sf::Color::Blue);
			line[1] = sf::Vertex(static_cast<sf::Vector2f>(testFormatedData[n+1]),sf::Color::Blue);
			myWin_->draw(line,2,sf::Lines);
		}

		myWin_->display();
	}
}

//--------------------------------------
fChart::fChart() : fWidget()
{
	rect_.setSize(static_cast<sf::Vector2f>(size_));
	rect_.setFillColor(sf::Color::White);
	rect_.setOutlineColor(sf::Color::Black);
	rect_.setOutlineThickness(2);
}

//--------------------------------------
void fChart::setPosition(sf::Vector2f& parPos)
{
	pos_ = static_cast<sf::Vector2u>(parPos);
	rect_.setPosition(parPos);
}

//--------------------------------------
sf::Vector2f fChart::getPosition() const
{
	return static_cast<sf::Vector2f>(pos_);
}

//--------------------------------------
void fChart::setSize(sf::Vector2f& parSize)
{
	size_ = static_cast<sf::Vector2u>(parSize);
	rect_.setSize(parSize);
}

//--------------------------------------
sf::Vector2f fChart::getSize() const
{
	return static_cast<sf::Vector2f>(size_);
}

//--------------------------------------
sf::RectangleShape& fChart::renderBackGround()
{
	return rect_;
}

//--------------------------------------
std::vector<sf::Vector2u>& fChart::renderData(std::vector<double> & parData)
{
	sf::Vector2u tmp;
	data_ = parData;
	formatedData_.clear();
	double min = *std::min_element(data_.begin(), data_.end());
	double max = *std::max_element(data_.begin(), data_.end());
	double pixelX = rect_.getSize().x / data_.size();
	double pixelY = rect_.getSize().y / ((max) - (min));
	
	for (unsigned n = 0; n < data_.size(); ++n) {
		tmp.x = n * pixelX + pos_.x;
		tmp.y = ((max) - data_[n]) * pixelY + pos_.y;
		formatedData_.push_back(tmp);
	}
	
	return formatedData_;
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


























