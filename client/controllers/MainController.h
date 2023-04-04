//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_MAINCONTROLLER_H
#define CLIENT_MAINCONTROLLER_H


#include "../models/CanvasModel.h"
#include "../models/GeneratedImagesModel.h"

class MainController {
public:
    MainController(CanvasModel* canvasModel, GeneratedImagesModel* generatedImagesModel);
public slots:
    void generateImages();
    void generateShadow();
private:
    CanvasModel* _canvasModel;
    GeneratedImagesModel* _generatedImagesModel;

};


#endif //CLIENT_MAINCONTROLLER_H
