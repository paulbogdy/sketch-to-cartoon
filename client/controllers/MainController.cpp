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
    auto images = Settings::getInstance().getGenerativeStrategy()->generateFromSketch(_canvasModel->image());
    _generatedImagesModel->setImages(images);
}
