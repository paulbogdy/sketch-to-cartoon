//
// Created by maneologu on 02.04.2023.
//

#include "Settings.h"
#include "../strategies/generative_strategies/DefaultStrategy.h"
#include "../strategies/generative_strategies/SketchyGanStrategy.h"

std::shared_ptr<Settings> Settings::_instance = nullptr;
std::mutex Settings::_mutex;

Settings::Settings(QObject* parent)
        : QObject(parent), _generativeStrategy(new SketchyGanStrategy()) {
    // Initialize your settings here
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

void Settings::setGenerativeStrategy(GenerativeStrategy *strategy) {
    delete _generativeStrategy;
    _generativeStrategy = strategy;
}

GenerativeStrategy *Settings::getGenerativeStrategy() {
    return _generativeStrategy;
}
