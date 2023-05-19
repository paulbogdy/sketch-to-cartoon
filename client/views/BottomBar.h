//
// Created by maneologu on 03.05.2023.
//

#ifndef CLIENT_BOTTOMBAR_H
#define CLIENT_BOTTOMBAR_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QHBoxLayout>
#include "../models/MessageBus.h"


#include <QStatusBar>
#include <QTimer>

class BottomBar : public QStatusBar {
Q_OBJECT

public:
    BottomBar(QWidget *parent = nullptr, QString textColor="rgb(255, 255, 255)");

public slots:
    void displayLoadingMessage(const QString &message, bool isLoading);

private:
    QLabel *loadingLabel;
    QString baseMessage;
    int dotCount;
    QTimer *timer;
};


#endif //CLIENT_BOTTOMBAR_H
