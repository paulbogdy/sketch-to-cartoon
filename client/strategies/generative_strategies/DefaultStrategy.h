//
// Created by maneologu on 03.04.2023.
//

#ifndef CLIENT_DEFAULTSTRATEGY_H
#define CLIENT_DEFAULTSTRATEGY_H

#include "GenerativeStrategy.h"

class DefaultStrategy: public GenerativeStrategy {
    Q_OBJECT
public:
    explicit DefaultStrategy(QObject* parent = nullptr);
    QVector<QImage> generateFromSketch(QImage sketch, int numSamples) override;
    QImage generateShadow(QImage sketch, int numSamples) override;
};


#endif //CLIENT_DEFAULTSTRATEGY_H
