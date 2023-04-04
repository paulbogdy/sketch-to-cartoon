//
// Created by maneologu on 03.04.2023.
//

#ifndef CLIENT_GENERATIVESTRATEGY_H
#define CLIENT_GENERATIVESTRATEGY_H


#include <QImage>

class GenerativeStrategy {
public:
    virtual QVector<QImage> generateFromSketch(QImage sketch) = 0;
    virtual ~GenerativeStrategy() = default;
};


#endif //CLIENT_GENERATIVESTRATEGY_H
