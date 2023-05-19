//
// Created by maneologu on 03.05.2023.
//

#ifndef CLIENT_GENERATIVESTRATEGYFACTORY_H
#define CLIENT_GENERATIVESTRATEGYFACTORY_H

#include <QObject>
#include "GenerativeStrategy.h"
#include "GanInversionStrategy.h"
#include "DefaultStrategy.h"

class GenerativeStrategyFactory {
public:
    static GenerativeStrategy* createStrategy(const QString& strategyName, QObject* parent = nullptr) {
        if (strategyName == "Gan Inversion") {
            return new GanInversionStrategy(parent);
        } else if (strategyName == "Default") {
            return new DefaultStrategy(parent);
        } else {
            throw std::invalid_argument("Invalid strategy name");
        }
    }
};



#endif //CLIENT_GENERATIVESTRATEGYFACTORY_H
