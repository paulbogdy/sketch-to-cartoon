//
// Created by maneologu on 26.04.2023.
//

#ifndef CLIENT_CANVASCONTROLLER_H
#define CLIENT_CANVASCONTROLLER_H

#include <QObject>
#include "../models/CanvasModel.h"
#include "../views/CanvasView.h"

class CanvasController: public QObject {
    Q_OBJECT

public:
    explicit CanvasController(CanvasModel* model, CanvasView* view, QObject* parent = nullptr);

public slots:
    void generateShadow();
    void undo();
    void redo();
    void saveSketch();
    void loadSketch();

    void selectPen();
    void selectEraser();

private:
    CanvasModel* _canvasModel;
    CanvasView* _canvasView;

};


#endif //CLIENT_CANVASCONTROLLER_H
