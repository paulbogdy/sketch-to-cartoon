//
// Created by maneologu on 03.04.2023.
//

#ifndef CLIENT_DEFAULTSTRATEGY_H
#define CLIENT_DEFAULTSTRATEGY_H

#include "GenerativeStrategy.h"

class DefaultStrategy: public GenerativeStrategy {
public:
    QVector<QImage> generateFromSketch(QImage sketch) override;
    ~DefaultStrategy() override;
};


#endif //CLIENT_DEFAULTSTRATEGY_H
