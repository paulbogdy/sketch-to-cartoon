//
// Created by maneologu on 03.05.2023.
//

#include "BottomBar.h"

BottomBar::BottomBar(QWidget *parent, QString textColor) : QStatusBar(parent), dotCount(0) {
    loadingLabel = new QLabel(this);
    loadingLabel->setStyleSheet("QLabel { color : " + textColor + "; }");
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this]() {
        dotCount = (dotCount + 1) % 4;
        loadingLabel->setText(baseMessage + QString(dotCount, '.'));
    });
    addWidget(loadingLabel);
}

void BottomBar::displayLoadingMessage(const QString &message, bool isLoading) {
    baseMessage = message;
    loadingLabel->setText(baseMessage);
    if (isLoading) {
        timer->start(500); // adjust the interval to your preference
    } else {
        timer->stop();
    }
}