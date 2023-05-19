//
// Created by maneologu on 26.04.2023.
//

#include "GeneratedImagesController.h"
#include "../settings/Settings.h"
#include <QtConcurrent/QtConcurrentRun>
#include <QFutureWatcher>
#include "../models/MessageBus.h"

GeneratedImagesController::GeneratedImagesController(
        CanvasModel *canvasModel,
        GeneratedImagesModel *generatedImagesModel,
        GeneratedImagesView *generatedImagesView,
        QObject *parent):
        QObject(parent)
        , _canvasModel(canvasModel)
        , _generatedImagesModel(generatedImagesModel)
        , _generatedImagesView(generatedImagesView) {
}

void GeneratedImagesController::generateImages() {
    MessageBus::getInstance().publishMessage("Generating images", true);
    auto strategy = Settings::getInstance().getGenerativeStrategy();
    auto sketch = _canvasModel->image();
    auto numSamples = Settings::getInstance().getImagesToGenerate();

    QFutureWatcher<QVector<QImage>> *watcher = new QFutureWatcher<QVector<QImage>>(this);
    connect(watcher, &QFutureWatcher<QVector<QImage>>::finished, this, [this, watcher]() {
        _generatedImagesModel->setImages(watcher->result());
        MessageBus::getInstance().publishMessage("Images generated", false);
        watcher->deleteLater();
    });

    QFuture<QVector<QImage>> future = QtConcurrent::run([strategy, sketch, numSamples]() {
        return strategy->generateFromSketch(sketch, numSamples);
    });

    watcher->setFuture(future);
}
