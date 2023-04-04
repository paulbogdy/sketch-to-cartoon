//
// Created by maneologu on 01.04.2023.
//

#ifndef CLIENT_CANVASVIEW_H
#define CLIENT_CANVASVIEW_H


#include <QWidget>
#include <QPoint>
#include <QPen>
#include "../models/CanvasModel.h"


class CanvasView : public QWidget {
Q_OBJECT

public:
    explicit CanvasView(CanvasModel* canvasModel, QWidget* parent = nullptr);

    void setImage(const QImage& image);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void drawLineTo(const QPoint& endPoint);

    CanvasModel* _canvasModel;
    QImage _image;
    QPoint _lastPoint;
    QPen _pen;
};



#endif //CLIENT_CANVASVIEW_H
