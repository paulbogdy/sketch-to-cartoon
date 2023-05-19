//
// Created by maneologu on 03.04.2023.
//

#ifndef CLIENT_SKETCHYGANSTRATEGY_H
#define CLIENT_SKETCHYGANSTRATEGY_H


#include <QImage>
#pragma push_macro("slots")
#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#pragma pop_macro("slots")
#include "GenerativeStrategy.h"

class SketchyGanStrategy: public GenerativeStrategy {
    Q_OBJECT
public:
    explicit SketchyGanStrategy(QObject* parent = nullptr);
    QVector<QImage> generateFromSketch(QImage sketch, int numSamples) override;
private:
    torch::jit::script::Module _generator;
    torch::Device _device;
};


#endif //CLIENT_SKETCHYGANSTRATEGY_H
