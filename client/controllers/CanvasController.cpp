//
// Created by maneologu on 26.04.2023.
//

#include "CanvasController.h"
#include <QDebug>
#include "../settings/Settings.h"
#include "../models/MessageBus.h"
#include "../drawing_tools/PenTool.h"
#include "../drawing_tools/EraserTool.h"
#include <QtConcurrent/QtConcurrentRun>
#include <QFutureWatcher>
#include <QFileDialog>

CanvasController::CanvasController(CanvasModel *model, CanvasView *view, QObject *parent)
        : QObject(parent), _canvasModel(model), _canvasView(view) {
    connect(_canvasView, &CanvasView::imageChanged, _canvasModel, &CanvasModel::setImage);
    connect(_canvasView, &CanvasView::undoRequest, _canvasModel, &CanvasModel::undo);
    connect(_canvasView, &CanvasView::redoRequest, _canvasModel, &CanvasModel::redo);
    connect(_canvasModel, &CanvasModel::shadowChanged, _canvasView, &CanvasView::updateShadow);
    connect(_canvasModel, &CanvasModel::imageChanged, _canvasView, &CanvasView::updateImage);
    connect(_canvasView, &CanvasView::beginDraw, _canvasModel, &CanvasModel::beginDraw);
    connect(_canvasView, &CanvasView::draw, _canvasModel, &CanvasModel::draw);
    connect(_canvasView, &CanvasView::endDraw, _canvasModel, &CanvasModel::endDraw);
}

void CanvasController::generateShadow() {
    MessageBus::getInstance().publishMessage("Generating shadow", true);
    auto strategy = Settings::getInstance().getGenerativeStrategy();
    auto sketch = _canvasModel->image();
    auto numSamples = Settings::getInstance().getImagesForShadowDraw();

    QFutureWatcher<QImage> *watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QVector<QImage>>::finished, this, [this, watcher]() {
        _canvasModel->setShadow(watcher->result());
        MessageBus::getInstance().publishMessage("Shadow generated", false);
        watcher->deleteLater();
    });

    QFuture<QImage> future = QtConcurrent::run([strategy, sketch, numSamples]() {
        return strategy->generateShadow(sketch, numSamples);
    });

    watcher->setFuture(future);
}

void CanvasController::undo() {
    qDebug() << "Undoing...";
    _canvasModel->undo();
}

void CanvasController::redo() {
    qDebug() << "Redoing...";
    _canvasModel->redo();
}

void CanvasController::saveSketch() {
    if(_canvasModel->image().isNull()) {
        MessageBus::getInstance().publishMessage("Sketch is empty", false);
        return;
    } else {
        MessageBus::getInstance().publishMessage("Saving sketch", true);
        auto image = _canvasModel->image();
        QString selectedFilter;
        auto fileName = QFileDialog::getSaveFileName(nullptr, "Save sketch", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)", &selectedFilter);

        if(fileName.isEmpty()) {
            MessageBus::getInstance().publishMessage("Saving cancelled", false);
            return;
        } else {
            QFileInfo info(fileName);
            QString fileExtension = info.suffix().toLower();
            QStringList supportedExtensions = {"png", "jpg", "jpeg"};

            if (!supportedExtensions.contains(fileExtension)) {
                if (selectedFilter.startsWith("PNG")) {
                    fileName += ".png";
                    fileExtension = "png";
                } else if (selectedFilter.startsWith("JPEG")) {
                    fileName += ".jpg";
                    fileExtension = "jpg";
                }
            }

            image.save(fileName, fileExtension.toUpper().toUtf8().constData());
            MessageBus::getInstance().publishMessage("Sketch saved", false);
        }
    }
}

void CanvasController::loadSketch() {
    MessageBus::getInstance().publishMessage("Loading sketch", true);
    auto fileName = QFileDialog::getOpenFileName(nullptr, "Load sketch", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)");
    if(fileName.isEmpty()) {
        MessageBus::getInstance().publishMessage("Loading cancelled", false);
        return;
    } else {
        QImage image;
        image.load(fileName);
        _canvasModel->setImage(image);
        MessageBus::getInstance().publishMessage("Sketch loaded", false);
    }
}

void CanvasController::selectPen() {
    _canvasModel->setDrawTool(std::make_shared<PenTool>());
    _canvasModel->setDrawToolWidth(_drawToolWidth);
}

void CanvasController::selectEraser() {
    _canvasModel->setDrawTool(std::make_shared<EraserTool>());
    _canvasModel->setDrawToolWidth(_drawToolWidth);
}

void CanvasController::selectDrawToolWidth(int width) {
    _drawToolWidth = width;
    _canvasModel->setDrawToolWidth(width);
}
