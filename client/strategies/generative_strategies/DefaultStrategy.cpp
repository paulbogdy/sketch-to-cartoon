//
// Created by maneologu on 03.04.2023.
//

#include "DefaultStrategy.h"

QVector<QImage> DefaultStrategy::generateFromSketch(QImage sketch) {
    QVector<QImage> images;
    for(int i=0; i<8; i++) {
        images.push_back(sketch);
    }
    return images;
}

DefaultStrategy::~DefaultStrategy() {};
