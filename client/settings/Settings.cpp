//
// Created by maneologu on 02.04.2023.
//

#include "Settings.h"
#include "../strategies/generative_strategies/DefaultStrategy.h"
#include "../strategies/generative_strategies/SketchyGanStrategy.h"
#include "../strategies/generative_strategies/GanInversionStrategy.h"
#include "../strategies/generative_strategies/GenerativeStrategyFactory.h"

std::shared_ptr<Settings> Settings::_instance = nullptr;
std::mutex Settings::_mutex;

Settings::Settings(QObject* parent)
        : QObject(parent), _strategyName("Default") {
    _generativeStrategy = GenerativeStrategyFactory::createStrategy(_strategyName);
}

Settings::~Settings() {
    delete _generativeStrategy;
}

Settings& Settings::getInstance() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_instance) {
        _instance = std::shared_ptr<Settings>(new Settings());
    }
    return *_instance;
}

void Settings::setGenerativeStrategy(QString strategyName) {
    qDebug() << "Settings::setGenerativeStrategy: " << strategyName;
    delete _generativeStrategy;
    _strategyName = strategyName;
    _generativeStrategy = GenerativeStrategyFactory::createStrategy(_strategyName) ;
}

GenerativeStrategy *Settings::getGenerativeStrategy() {
    return _generativeStrategy;
}

void Settings::setImagesToGenerate(int imagesToGenerate) {
    _imagesToGenerate = imagesToGenerate;
}

int Settings::getImagesToGenerate() const {
    return _imagesToGenerate;
}

void Settings::setImagesForShadowDraw(int imagesForShadowDraw) {
    _imagesForShadowDraw = imagesForShadowDraw;
}

int Settings::getImagesForShadowDraw() const {
    return _imagesForShadowDraw;
}

QString Settings::getGenerativeStrategyName() const {
    return _strategyName;
}
