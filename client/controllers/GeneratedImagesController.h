//
// Created by maneologu on 26.04.2023.
//

#ifndef CLIENT_GENERATEDIMAGESCONTROLLER_H
#define CLIENT_GENERATEDIMAGESCONTROLLER_H

#include <QObject>
#include "../models/GeneratedImagesModel.h"
#include "../views/GeneratedImagesView.h"
#include "../models/CanvasModel.h"

class GeneratedImagesController: public QObject {

    Q_OBJECT

public:
    explicit GeneratedImagesController(CanvasModel* canvasModel,
                                       GeneratedImagesModel* generatedImagesModel,
                                       GeneratedImagesView* generatedImagesView,
                                       QObject* parent = nullptr);
public slots:
    void generateImages();
private:
    GeneratedImagesModel* _generatedImagesModel;
    GeneratedImagesView* _generatedImagesView;
    CanvasModel* _canvasModel;

};


#endif //CLIENT_GENERATEDIMAGESCONTROLLER_H
