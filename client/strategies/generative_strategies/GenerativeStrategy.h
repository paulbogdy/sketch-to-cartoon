//
// Created by maneologu on 03.04.2023.
//

#ifndef CLIENT_GENERATIVESTRATEGY_H
#define CLIENT_GENERATIVESTRATEGY_H


#include <QImage>

class GenerativeStrategy: public QObject {
    Q_OBJECT
public:
    explicit GenerativeStrategy(QObject* parent = nullptr);
    virtual QVector<QImage> generateFromSketch(QImage sketch, int numSamples) = 0;
    virtual QImage generateShadow(QImage sketch, int numSamples) = 0;
};


#endif //CLIENT_GENERATIVESTRATEGY_H
