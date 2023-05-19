//
// Created by maneologu on 19.04.2023.
//

#ifndef CLIENT_GANINVERSIONSTRATEGY_H
#define CLIENT_GANINVERSIONSTRATEGY_H

#include "GenerativeStrategy.h"
#include <QtNetwork/QLocalSocket>
#include <QByteArray>
#include <QDataStream>
#include <QProcess>

class GanInversionStrategy: public GenerativeStrategy {
    Q_OBJECT
public:
    enum TaskType {
        GENERATE = 0,
        SHADOW = 1
    };
    explicit GanInversionStrategy(QObject* parent = nullptr);
    QVector<QImage> generateFromSketch(QImage sketch, int numSamples) override;
    QImage generateShadow(QImage sketch, int numSamples) override;
protected:
    void connectToServer(QLocalSocket& _socket);
private:

    void disconnectFromServer(QLocalSocket& _socket);

    void sendTaskType(QLocalSocket& _socket, QDataStream &stream, TaskType type);

    void sendSketchData(QLocalSocket& _socket, QDataStream &stream, QImage &sketch);

    QImage retrieveImage(QLocalSocket& _socket);

    void sendShadowComplexity(QLocalSocket& _socket, QDataStream &stream, int numSamples);

    void sendNumberOfRequestedImages(QLocalSocket &_socket, QDataStream &stream, int numSamples);
};


#endif //CLIENT_GANINVERSIONSTRATEGY_H
