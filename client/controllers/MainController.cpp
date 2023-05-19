//
// Created by maneologu on 02.04.2023.
//

#include "MainController.h"
#include "../settings/Settings.h"

MainController::MainController(CanvasModel *canvasModel, GeneratedImagesModel *generatedImagesModel)
    : _canvasModel(canvasModel)
    , _generatedImagesModel(generatedImagesModel)
    {
}

void MainController::generateImages() {
    auto images = Settings::getInstance().getGenerativeStrategy()->generateFromSketch(_canvasModel->image(), 0);
    _generatedImagesModel->setImages(images);
}

void MainController::generateShadow() {
    auto shadow = Settings::getInstance().getGenerativeStrategy()->generateShadow(_canvasModel->image(), 0);
    _canvasModel->setShadow(shadow);
}
